"""POST /ingest: single image ingestion."""

import json
import logging
import time
import uuid

from fastapi import File, Form, HTTPException, UploadFile, status
from fastapi.responses import ORJSONResponse

from src.core.dependencies import VisualSearchDep
from src.routers.ingest._router import router
from src.routers.ingest.models import IndexedCounts, IngestResponse, NearDuplicateInfo, OCRInfo


logger = logging.getLogger(__name__)


@router.post(
    '',
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {'description': 'Image successfully ingested'},
        200: {'description': 'Image already exists (duplicate)'},
        400: {'description': 'Invalid input'},
        500: {'description': 'Internal server error'},
    },
    summary='Ingest single image',
    description="""
Ingest a single image with automatic indexing to appropriate OpenSearch indexes.

**Pipeline:**
1. Compute imohash for exact duplicate detection
2. Check if image already exists (if skip_duplicates=True)
3. Run YOLO detection + MobileCLIP embedding extraction
4. Run SCRFD face detection + ArcFace embedding extraction
5. Index to appropriate indexes:
   - Global embedding -> visual_search_global
   - Person detections -> visual_search_people
   - Vehicle detections -> visual_search_vehicles
   - Face embeddings -> visual_search_faces
6. Check for near-duplicates and assign to groups (if enabled)
7. Run OCR and index text content (if enabled)

**Duplicate Detection:**
- Uses imohash (fast perceptual hash) for exact duplicates
- Near-duplicate grouping uses CLIP embedding similarity (default threshold: 0.99)

**High Concurrency:**
- Uses CPU preprocessing for stability at high request rates
- Parallel inference for YOLO + CLIP + face detection
""",
)
async def ingest_image(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    image_id: str | None = Form(
        None,
        description='Unique image identifier. Auto-generated UUID if not provided.',
    ),
    image_path: str | None = Form(
        None,
        description='Original file path for retrieval. Defaults to image_id.',
    ),
    metadata: str | None = Form(
        None,
        description='JSON string of additional metadata to store with the image.',
    ),
    skip_duplicates: bool = Form(
        True,
        description='Skip processing if image hash already exists in index.',
    ),
    detect_near_duplicates: bool = Form(
        True,
        description='Check for visually similar images and assign to duplicate groups.',
    ),
    near_duplicate_threshold: float = Form(
        0.99,
        ge=0.90,
        le=1.0,
        description='Similarity threshold for near-duplicate grouping. '
        '0.99 matches near-identical images, 0.90 matches similar content.',
    ),
    enable_ocr: bool = Form(
        True,
        description='Run OCR to extract and index text content from the image.',
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
):
    """
    Ingest a single image with automatic multi-index routing.

    Supports high concurrency (300+ RPS) with CPU preprocessing.
    """

    start_time = time.perf_counter()

    try:
        # Read image bytes
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Empty image file',
            )

        # Generate image_id if not provided
        actual_image_id = image_id or str(uuid.uuid4())

        # Parse metadata JSON if provided
        parsed_metadata = None
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
                if not isinstance(parsed_metadata, dict):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail='Metadata must be a JSON object',
                    )
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Invalid metadata JSON: {e}',
                ) from e

        # Call service layer
        result = await search_service.ingest_image(
            image_bytes=image_bytes,
            image_id=actual_image_id,
            image_path=image_path,
            metadata=parsed_metadata,
            skip_duplicates=skip_duplicates,
            detect_near_duplicates=detect_near_duplicates,
            near_duplicate_threshold=near_duplicate_threshold,
            enable_ocr=enable_ocr,
            enable_detection=enable_detection,
            enable_faces=enable_faces,
            enable_clip=enable_clip,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Build response based on result status
        if result.get('status') == 'duplicate':
            return ORJSONResponse(
                status_code=status.HTTP_200_OK,
                content=IngestResponse(
                    status='duplicate',
                    image_id=result['image_id'],
                    imohash=result.get('imohash'),
                    existing_image_id=result.get('existing_image_id'),
                    existing_image_path=result.get('existing_image_path'),
                    message=result.get('message', 'Image already exists in index'),
                    total_time_ms=round(elapsed_ms, 2),
                ).model_dump(by_alias=True, exclude_none=True),
            )

        if result.get('status') == 'error':
            return IngestResponse(
                status='error',
                image_id=result['image_id'],
                error=result.get('error', 'Unknown error'),
                total_time_ms=round(elapsed_ms, 2),
            )

        # Success response
        indexed_data = result.get('indexed', {})
        indexed = IndexedCounts(
            **{
                'global': indexed_data.get('global', False),
                'vehicles': indexed_data.get('vehicles', 0),
                'people': indexed_data.get('people', 0),
                'faces': indexed_data.get('faces', 0),
            }
        )

        near_duplicate = None
        if result.get('near_duplicate'):
            nd = result['near_duplicate']
            near_duplicate = NearDuplicateInfo(
                action=nd['action'],
                group_id=nd['group_id'],
                similarity=nd['similarity'],
                matched_image=nd['matched_image'],
            )

        ocr_info = None
        if result.get('ocr'):
            ocr_data = result['ocr']
            ocr_info = OCRInfo(
                num_texts=ocr_data.get('num_texts', 0),
                full_text=ocr_data.get('full_text', ''),
                indexed=ocr_data.get('indexed', False),
            )

        return IngestResponse(
            status='success',
            image_id=result['image_id'],
            num_detections=result.get('num_detections', 0),
            num_faces=result.get('num_faces', 0),
            embedding_norm=result.get('embedding_norm', 0.0),
            imohash=result.get('imohash'),
            indexed=indexed,
            near_duplicate=near_duplicate,
            ocr=ocr_info,
            errors=result.get('errors') if result.get('errors') else None,
            total_time_ms=round(elapsed_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Image ingestion failed: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Image ingestion failed: {e!s}',
        ) from e
