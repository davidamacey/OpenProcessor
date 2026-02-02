"""
Data Ingestion Router.

Provides endpoints for ingesting images into OpenSearch with automatic
indexing to appropriate categories (global, vehicles, people, faces, ocr).

Endpoints:
- POST /ingest - Single image ingestion (high concurrency support)
- POST /ingest/batch - Batch ingest (up to 64 images)
- POST /ingest/directory - Bulk load from directory path on server

Features:
- Duplicate detection via imohash
- Near-duplicate grouping via CLIP embeddings
- Auto-routing to category indexes based on detections
- OCR text extraction and indexing
- Face detection and ArcFace embedding indexing
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.core.dependencies import VisualSearchDep


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/ingest',
    tags=['Data Ingestion'],
    default_response_class=ORJSONResponse,
)


# =============================================================================
# Response Models
# =============================================================================


class IndexedCounts(BaseModel):
    """Counts of documents indexed per category."""

    global_: bool = Field(..., alias='global', description='Whether global embedding was indexed')
    vehicles: int = Field(default=0, description='Number of vehicle detections indexed')
    people: int = Field(default=0, description='Number of person detections indexed')
    faces: int = Field(default=0, description='Number of faces indexed')

    class Config:
        populate_by_name = True


class NearDuplicateInfo(BaseModel):
    """Information about near-duplicate detection result."""

    action: Literal['joined_group', 'created_group'] = Field(
        ..., description='Action taken for duplicate grouping'
    )
    group_id: str = Field(..., description='Duplicate group ID')
    similarity: float = Field(
        ..., ge=0.0, le=1.0, description='Similarity score with matched image'
    )
    matched_image: str = Field(..., description='Image ID of the matched duplicate')


class OCRInfo(BaseModel):
    """OCR processing result information."""

    num_texts: int = Field(..., ge=0, description='Number of text regions detected')
    full_text: str = Field(..., description='Concatenated extracted text (truncated to 200 chars)')
    indexed: bool = Field(..., description='Whether OCR results were indexed')


class IngestResponse(BaseModel):
    """
    Response for single image ingestion.

    Includes status, indexing results, and optional duplicate/OCR information.
    """

    status: Literal['success', 'duplicate', 'error'] = Field(..., description='Ingestion status')
    image_id: str = Field(..., description='Image identifier')
    num_detections: int = Field(default=0, ge=0, description='Number of YOLO detections')
    num_faces: int = Field(default=0, ge=0, description='Number of faces detected')
    embedding_norm: float = Field(default=0.0, description='L2 norm of global embedding')
    imohash: str | None = Field(
        default=None, description='Image content hash for duplicate detection'
    )
    indexed: IndexedCounts | None = Field(
        default=None, description='Counts of documents indexed per category'
    )
    near_duplicate: NearDuplicateInfo | None = Field(
        default=None, description='Near-duplicate grouping result (if detected)'
    )
    ocr: OCRInfo | None = Field(default=None, description='OCR processing result (if enabled)')
    existing_image_id: str | None = Field(
        default=None, description='ID of existing image (if duplicate)'
    )
    existing_image_path: str | None = Field(
        default=None, description='Path of existing image (if duplicate)'
    )
    message: str | None = Field(default=None, description='Additional status message')
    error: str | None = Field(default=None, description='Error message (if status is error)')
    errors: list[str] | None = Field(default=None, description='List of non-fatal errors')
    total_time_ms: float | None = Field(default=None, description='Processing time in milliseconds')


class BatchIndexedCounts(BaseModel):
    """Aggregate counts for batch ingestion."""

    global_: int = Field(
        default=0, alias='global', description='Number of global embeddings indexed'
    )
    vehicles: int = Field(default=0, description='Number of vehicle detections indexed')
    people: int = Field(default=0, description='Number of person detections indexed')
    faces: int = Field(default=0, description='Number of faces indexed')
    ocr: int = Field(default=0, description='Number of images with OCR indexed')

    class Config:
        populate_by_name = True


class DuplicateDetail(BaseModel):
    """Details about a detected duplicate."""

    image_id: str = Field(..., description='ID of the duplicate image')
    existing_image_id: str | None = Field(default=None, description='ID of existing image in index')
    imohash: str = Field(..., description='Content hash')


class ErrorDetail(BaseModel):
    """Details about an ingestion error."""

    image_id: str = Field(..., description='ID of the failed image')
    error: str = Field(..., description='Error message')


class NearDuplicateDetail(BaseModel):
    """Details about a near-duplicate assignment."""

    image_id: str = Field(..., description='ID of the image')
    action: str = Field(..., description='Action taken (joined_group or created_group)')
    group_id: str = Field(..., description='Duplicate group ID')
    similarity: float = Field(..., description='Similarity score')
    matched_image: str = Field(..., description='ID of matched image')


class BatchIngestResponse(BaseModel):
    """
    Response for batch image ingestion.

    Includes summary statistics and detailed lists of duplicates, errors,
    and near-duplicate assignments.
    """

    status: Literal['success', 'partial', 'error'] = Field(..., description='Overall batch status')
    total: int = Field(..., ge=0, description='Total images submitted')
    processed: int = Field(default=0, ge=0, description='Successfully processed images')
    duplicates: int = Field(default=0, ge=0, description='Exact duplicates skipped')
    errors_count: int = Field(default=0, ge=0, description='Failed images count')
    indexed: BatchIndexedCounts = Field(
        default_factory=BatchIndexedCounts, description='Aggregate indexed counts'
    )
    near_duplicates: int = Field(default=0, ge=0, description='Images assigned to duplicate groups')
    deferred_ops: int = Field(
        default=0, ge=0, description='Images with deferred near-duplicate detection'
    )
    duplicate_details: list[DuplicateDetail] | None = Field(
        default=None, description='Details of detected duplicates'
    )
    error_details: list[ErrorDetail] | None = Field(
        default=None, description='Details of failed images'
    )
    near_duplicate_details: list[NearDuplicateDetail] | None = Field(
        default=None, description='Details of near-duplicate assignments'
    )
    total_time_ms: float | None = Field(default=None, description='Total batch processing time')


class DirectoryIngestResponse(BaseModel):
    """
    Response for directory bulk ingestion.

    Extends batch response with directory-specific metadata.
    """

    status: Literal['success', 'partial', 'error'] = Field(
        ..., description='Overall ingestion status'
    )
    directory: str = Field(..., description='Source directory path')
    total_files: int = Field(..., ge=0, description='Total image files found')
    total: int = Field(..., ge=0, description='Total images submitted for processing')
    processed: int = Field(default=0, ge=0, description='Successfully processed images')
    duplicates: int = Field(default=0, ge=0, description='Exact duplicates skipped')
    errors_count: int = Field(default=0, ge=0, description='Failed images count')
    indexed: BatchIndexedCounts = Field(
        default_factory=BatchIndexedCounts, description='Aggregate indexed counts'
    )
    near_duplicates: int = Field(default=0, ge=0, description='Images assigned to duplicate groups')
    skipped_extensions: list[str] | None = Field(
        default=None, description='File extensions that were skipped'
    )
    total_time_ms: float | None = Field(default=None, description='Total processing time')
    error: str | None = Field(default=None, description='Error message if status is error')


# =============================================================================
# Endpoints
# =============================================================================


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
    import time

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
    import time

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


@router.post(
    '/directory',
    response_model=DirectoryIngestResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {'description': 'Directory successfully processed'},
        400: {'description': 'Invalid input or directory not found'},
        500: {'description': 'Internal server error'},
    },
    summary='Bulk ingest from directory',
    description="""
Bulk ingest all images from a server-side directory path.

**Supported formats:** JPEG, PNG, WebP, BMP, TIFF

**Use case:** Initial library ingestion, server-side batch processing.

**Note:** This endpoint reads files from the server filesystem.
Ensure the directory path is accessible from within the container.

**Processing:**
- Files are processed in batches of 64 for optimal throughput
- Progress is logged to server logs
- Partial failures are allowed - check response for details
""",
)
async def ingest_directory(
    search_service: VisualSearchDep,
    directory: str = Query(
        ...,
        description='Absolute path to directory containing images',
    ),
    recursive: bool = Query(
        True,
        description='Recursively process subdirectories',
    ),
    max_images: int | None = Query(
        None,
        ge=1,
        le=100000,
        description='Maximum number of images to process (None = all)',
    ),
    batch_size: int = Query(
        64,
        ge=1,
        le=64,
        description='Batch size for processing (max 64)',
    ),
    skip_duplicates: bool = Query(
        True,
        description='Skip processing if image hash already exists',
    ),
    detect_near_duplicates: bool = Query(
        True,
        description='Check for near-duplicates and assign to groups',
    ),
    near_duplicate_threshold: float = Query(
        0.99,
        ge=0.90,
        le=1.0,
        description='Similarity threshold for near-duplicate grouping',
    ),
    enable_ocr: bool = Query(
        True,
        description='Run OCR on images',
    ),
    enable_detection: bool = Query(
        True,
        description='Run YOLO object detection',
    ),
    enable_faces: bool = Query(
        True,
        description='Run face detection and embedding extraction',
    ),
    enable_clip: bool = Query(
        True,
        description='Run MobileCLIP global image embedding',
    ),
    defer_heavy_ops: bool = Query(
        False,
        description='Defer heavy operations for faster bulk ingestion',
    ),
):
    """
    Bulk ingest all images from a server-side directory.

    Files are processed in configurable batches with progress logging.
    """
    import time

    start_time = time.perf_counter()

    try:
        # Validate directory exists
        dir_path = Path(directory)
        if not dir_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Directory not found: {directory}',
            )

        if not dir_path.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Path is not a directory: {directory}',
            )

        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}

        # Collect image files
        if recursive:
            all_files = list(dir_path.rglob('*'))
        else:
            all_files = list(dir_path.glob('*'))

        image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in image_extensions]

        # Track skipped extensions
        skipped_extensions = set()
        for f in all_files:
            if f.is_file() and f.suffix.lower() not in image_extensions:
                skipped_extensions.add(f.suffix.lower())

        if not image_files:
            return DirectoryIngestResponse(
                status='success',
                directory=directory,
                total_files=0,
                total=0,
                processed=0,
                duplicates=0,
                errors_count=0,
                indexed=BatchIndexedCounts(),
                near_duplicates=0,
                skipped_extensions=list(skipped_extensions) if skipped_extensions else None,
                total_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )

        # Apply max_images limit
        if max_images and len(image_files) > max_images:
            image_files = image_files[:max_images]

        total_files = len(image_files)
        logger.info(f'Starting directory ingestion: {total_files} images from {directory}')

        # Process in batches
        total_processed = 0
        total_duplicates = 0
        total_errors = 0
        total_near_duplicates = 0
        aggregate_indexed = {
            'global': 0,
            'vehicles': 0,
            'people': 0,
            'faces': 0,
            'ocr': 0,
        }

        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = image_files[batch_start:batch_end]

            # Read batch images
            images_data: list[tuple[bytes, str, str | None]] = []
            for file_path in batch_files:
                try:
                    image_bytes = file_path.read_bytes()
                    if not image_bytes:
                        continue

                    # Use relative path from directory as image_id
                    rel_path = file_path.relative_to(dir_path)
                    img_id = str(rel_path).replace(os.sep, '/')
                    img_path = str(file_path)

                    images_data.append((image_bytes, img_id, img_path))
                except Exception as e:
                    logger.warning(f'Failed to read {file_path}: {e}')
                    total_errors += 1

            if not images_data:
                continue

            # Process batch
            try:
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

                total_processed += result.get('processed', 0)
                total_duplicates += result.get('duplicates', 0)
                total_errors += result.get('errors_count', 0)
                total_near_duplicates += result.get('near_duplicates', 0)

                # Aggregate indexed counts
                indexed = result.get('indexed', {})
                aggregate_indexed['global'] += indexed.get('global', 0)
                aggregate_indexed['vehicles'] += indexed.get('vehicles', 0)
                aggregate_indexed['people'] += indexed.get('people', 0)
                aggregate_indexed['faces'] += indexed.get('faces', 0)
                aggregate_indexed['ocr'] += indexed.get('ocr', 0)

                logger.info(
                    f'Batch {batch_start // batch_size + 1}: '
                    f'processed={result.get("processed", 0)}, '
                    f'duplicates={result.get("duplicates", 0)}, '
                    f'errors={result.get("errors_count", 0)}'
                )

            except Exception as e:
                logger.error(f'Batch processing failed: {e}')
                total_errors += len(images_data)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Determine status
        status_value: Literal['success', 'partial', 'error'] = 'success'
        if total_errors > 0:
            if total_processed > 0:
                status_value = 'partial'
            else:
                status_value = 'error'

        logger.info(
            f'Directory ingestion complete: {total_processed} processed, '
            f'{total_duplicates} duplicates, {total_errors} errors in {elapsed_ms:.0f}ms'
        )

        return DirectoryIngestResponse(
            status=status_value,
            directory=directory,
            total_files=total_files,
            total=total_files,
            processed=total_processed,
            duplicates=total_duplicates,
            errors_count=total_errors,
            indexed=BatchIndexedCounts(**aggregate_indexed),
            near_duplicates=total_near_duplicates,
            skipped_extensions=list(skipped_extensions) if skipped_extensions else None,
            total_time_ms=round(elapsed_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Directory ingestion failed: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Directory ingestion failed: {e!s}',
        ) from e
