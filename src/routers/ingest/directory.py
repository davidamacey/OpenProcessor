"""POST /ingest/directory: bulk load from a directory path on the server."""

import logging
import os
import time
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, Query, status

from src.core.dependencies import VisualSearchDep
from src.routers.ingest._router import router
from src.routers.ingest.models import BatchIndexedCounts, DirectoryIngestResponse


logger = logging.getLogger(__name__)


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
