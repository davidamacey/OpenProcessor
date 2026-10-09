"""POST /ingest/batch: batch image ingestion (up to 64 images)."""

import json
import logging
import time
import uuid
from typing import Literal

from fastapi import File, Form, HTTPException, UploadFile, status

from src.core.dependencies import VisualSearchDep
from src.routers.ingest._router import router
from src.routers.ingest.models import (
    BatchIndexedCounts,
    BatchIngestResponse,
    DuplicateDetail,
    ErrorDetail,
    NearDuplicateDetail,
)


logger = logging.getLogger(__name__)


@router.post(
    '/batch',
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
- Parallel near-duplicate detection
- Reduced HTTP overhead

**Target throughput:** 300+ images/second with batch sizes of 32-64.

**High-Throughput Mode:** Set `defer_heavy_ops=true` to skip near-duplicate detection
during rapid ingestion. These can be processed later when system load is lower.

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
        description='Check for near-duplicates and assign to groups.',
    ),
    near_duplicate_threshold: float = Form(
        0.99,
        ge=0.90,
        le=1.0,
        description='Similarity threshold for near-duplicate grouping.',
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
        description='Run face detection (SCRFD) and embedding (ArcFace) extraction.',
    ),
    enable_clip: bool = Form(
        True,
        description='Run MobileCLIP to generate global image embedding.',
    ),
    defer_heavy_ops: bool = Form(
        False,
        description='Defer heavy operations (near-duplicate detection) for faster ingestion. '
        'Use during high-throughput bulk ingestion.',
    ),
):
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

        # Parse image_ids JSON array if provided
        parsed_ids: list[str | None] = [None] * len(images)
        if image_ids:
            try:
                ids_list = json.loads(image_ids)
                if not isinstance(ids_list, list):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail='image_ids must be a JSON array',
                    )
                if len(ids_list) != len(images):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f'image_ids length ({len(ids_list)}) must match images length ({len(images)})',
                    )
                parsed_ids = ids_list
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Invalid image_ids JSON: {e}',
                ) from e

        # Parse image_paths JSON array if provided
        parsed_paths: list[str | None] = [None] * len(images)
        if image_paths:
            try:
                paths_list = json.loads(image_paths)
                if not isinstance(paths_list, list):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail='image_paths must be a JSON array',
                    )
                if len(paths_list) != len(images):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f'image_paths length ({len(paths_list)}) must match images length ({len(images)})',
                    )
                parsed_paths = paths_list
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Invalid image_paths JSON: {e}',
                ) from e

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
            near_duplicate_threshold=near_duplicate_threshold,
            enable_ocr=enable_ocr,
            enable_detection=enable_detection,
            enable_faces=enable_faces,
            enable_clip=enable_clip,
            defer_heavy_ops=defer_heavy_ops,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Build response
        indexed_data = result.get('indexed', {})
        indexed = BatchIndexedCounts(
            **{
                'global': indexed_data.get('global', 0),
                'vehicles': indexed_data.get('vehicles', 0),
                'people': indexed_data.get('people', 0),
                'faces': indexed_data.get('faces', 0),
                'ocr': indexed_data.get('ocr', 0),
            }
        )

        # Parse duplicate details
        duplicate_details = None
        if result.get('duplicate_details'):
            duplicate_details = [
                DuplicateDetail(
                    image_id=d['image_id'],
                    existing_image_id=d.get('existing_image_id'),
                    imohash=d.get('imohash', ''),
                )
                for d in result['duplicate_details']
            ]

        # Parse error details
        error_details = None
        if result.get('error_details'):
            error_details = [
                ErrorDetail(
                    image_id=e['image_id'],
                    error=e.get('error', 'Unknown error'),
                )
                for e in result['error_details']
            ]

        # Parse near-duplicate details
        near_duplicate_details = None
        if result.get('near_duplicate_details'):
            near_duplicate_details = [
                NearDuplicateDetail(
                    image_id=nd['image_id'],
                    action=nd['action'],
                    group_id=nd['group_id'],
                    similarity=nd['similarity'],
                    matched_image=nd['matched_image'],
                )
                for nd in result['near_duplicate_details']
            ]

        # Determine overall status
        status_value: Literal['success', 'partial', 'error'] = 'success'
        if result.get('errors_count', 0) > 0:
            if result.get('processed', 0) > 0:
                status_value = 'partial'
            else:
                status_value = 'error'

        return BatchIngestResponse(
            status=status_value,
            total=result.get('total', len(images)),
            processed=result.get('processed', 0),
            duplicates=result.get('duplicates', 0),
            errors_count=result.get('errors_count', 0),
            indexed=indexed,
            near_duplicates=result.get('near_duplicates', 0),
            deferred_ops=result.get('deferred_ops', 0),
            duplicate_details=duplicate_details,
            error_details=error_details,
            near_duplicate_details=near_duplicate_details,
            total_time_ms=round(elapsed_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Batch ingestion failed: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Batch ingestion failed: {e!s}',
        ) from e
