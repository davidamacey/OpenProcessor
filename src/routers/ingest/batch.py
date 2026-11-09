"""POST /ingest/batch: batch image ingestion (up to 64 images)."""

import json
import logging
import time
import uuid
from typing import Any, Literal

from fastapi import File, Form, HTTPException, UploadFile, status

from src.core.dependencies import VisualSearchDep
from src.core.disk_space import DISK_GUARD
from src.routers._errors import server_error
from src.routers.ingest._router import router
from src.schemas.ingest import (
    BatchIndexedCounts,
    BatchIngestResponse,
    DuplicateDetail,
    ErrorDetail,
    ImageNearDuplicateCandidates,
    NearDuplicateCandidate,
)


logger = logging.getLogger(__name__)


def _json_list(raw: str | None, name: str, count: int) -> list[Any]:
    """The JSON array a form field carries, one entry per image; ``None`` per image when absent."""
    if not raw:
        return [None] * count
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f'Invalid {name} JSON: {e}'
        ) from e
    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f'{name} must be a JSON array'
        )
    if len(parsed) != count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'{name} length ({len(parsed)}) must match images length ({count})',
        )
    return parsed


def _overall_status(result: dict[str, Any]) -> Literal['success', 'partial', 'error']:
    if result.get('errors_count', 0) <= 0:
        return 'success'
    return 'partial' if result.get('processed', 0) > 0 else 'error'


def _response(result: dict[str, Any], submitted: int, elapsed_ms: float) -> BatchIngestResponse:
    """The wire body of one batch result."""
    indexed_data = result.get('indexed', {})
    indexed = BatchIndexedCounts(
        **{
            key: indexed_data.get(key, 0)
            for key in ('global', 'vehicles', 'people', 'faces', 'ocr')
        }
    )
    duplicate_details = [
        DuplicateDetail(
            image_id=d['image_id'],
            existing_image_id=d.get('existing_image_id'),
            imohash=d.get('imohash', ''),
        )
        for d in result.get('duplicate_details') or []
    ]
    error_details = [
        ErrorDetail(image_id=e['image_id'], error=e.get('error', 'Unknown error'))
        for e in result.get('error_details') or []
    ]
    candidates = [
        ImageNearDuplicateCandidates(
            image_id=entry['image_id'],
            candidates=[
                NearDuplicateCandidate(image_id=c['image_id'], cosine=c['cosine'])
                for c in entry['candidates']
            ],
        )
        for entry in result.get('near_duplicate_candidates') or []
    ]
    return BatchIngestResponse(
        status=_overall_status(result),
        total=result.get('total', submitted),
        processed=result.get('processed', 0),
        duplicates=result.get('duplicates', 0),
        errors_count=result.get('errors_count', 0),
        indexed=indexed,
        near_duplicates=result.get('near_duplicates', 0),
        deferred_ops=result.get('deferred_ops', 0),
        duplicate_details=duplicate_details or None,
        error_details=error_details or None,
        near_duplicate_candidates=candidates or None,
        total_time_ms=round(elapsed_ms, 2),
    )


@router.post(
    '/batch',
    dependencies=[DISK_GUARD],
    response_model=BatchIngestResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {'description': 'Batch successfully processed'},
        400: {'description': 'Invalid input'},
        500: {'description': 'Internal server error'},
    },
    summary='Batch ingest images',
    description="""
Batch ingest up to 64 images with optimized parallel processing.

**Performance:** 3-5x faster than individual /ingest calls.
- Parallel hash computation
- Batch duplicate checking (msearch - 10x faster than sequential)
- Batch Triton inference with dynamic batching
- Bulk OpenSearch indexing for images and faces
- Near-duplicate candidates listed in one query (the groups come from the `near_dup_core_group` job)
- Reduced HTTP overhead

**Target throughput:** 300+ images/second with batch sizes of 32-64.

**High-Throughput Mode:** Set `defer_heavy_ops=true` to skip the near-duplicate candidate lookup
during rapid ingestion; the groups are built by the job either way.

**Partial Failures:** The endpoint returns partial results if some images fail.
Check the `error_details` field for per-image error information.
""",
)
async def ingest_batch(
    search_service: VisualSearchDep,
    images: list[UploadFile] = File(..., description='Image files (JPEG/PNG), max 64'),
    image_ids: str | None = Form(
        None,
        description='JSON array of image IDs corresponding to images. Auto-generated if not provided.',
    ),
    image_paths: str | None = Form(
        None,
        description='JSON array of image paths corresponding to images.',
    ),
    skip_duplicates: bool = Form(
        True,
        description='Skip processing if image hash already exists.',
    ),
    detect_near_duplicates: bool = Form(
        True,
        description='Mark images for near-duplicate grouping and list their candidates.',
    ),
    enable_ocr: bool = Form(
        True,
        description='Run OCR on images (may reduce throughput).',
    ),
    enable_detection: bool = Form(
        True,
        description='Run YOLO object detection and index vehicle/person detections.',
    ),
    enable_faces: bool = Form(
        True,
        description='Run face detection (SCRFD) and embedding (ArcFace) extraction. '
        'Ignored (always off) when OP_FACES_ENABLED=false.',
    ),
    enable_clip: bool = Form(
        True,
        description='Run MobileCLIP to generate global image embedding.',
    ),
    defer_heavy_ops: bool = Form(
        False,
        description='Skip the near-duplicate candidate lookup for faster ingestion. '
        'Use during high-throughput bulk ingestion.',
    ),
) -> BatchIngestResponse:
    """
    Batch ingest multiple images with optimized parallel processing.

    Maximum 64 images per request for optimal GPU batching.
    """

    start_time = time.perf_counter()

    try:
        # Validate batch size
        if len(images) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='No images provided',
            )

        if len(images) > 64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Maximum 64 images per batch',
            )

        parsed_ids: list[str | None] = _json_list(image_ids, 'image_ids', len(images))
        parsed_paths: list[str | None] = _json_list(image_paths, 'image_paths', len(images))

        # Read all image bytes and prepare data tuples
        images_data: list[tuple[bytes, str, str | None]] = []
        for i, img in enumerate(images):
            image_bytes = await img.read()
            if not image_bytes:
                logger.warning(f'Empty image at index {i}: {img.filename}')
                continue

            img_id = parsed_ids[i] or str(uuid.uuid4())
            img_path = parsed_paths[i]
            images_data.append((image_bytes, img_id, img_path))

        if not images_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='All images are empty',
            )

        # Call service layer batch method
        result = await search_service.ingest_batch(
            images_data=images_data,
            skip_duplicates=skip_duplicates,
            detect_near_duplicates=detect_near_duplicates,
            enable_ocr=enable_ocr,
            enable_detection=enable_detection,
            enable_faces=enable_faces,
            enable_clip=enable_clip,
            defer_heavy_ops=defer_heavy_ops,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return _response(result, len(images), elapsed_ms)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Batch ingestion failed: {e}')
        raise server_error(500, 'Batch ingestion failed', e) from e
