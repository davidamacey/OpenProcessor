"""POST /ingest/directory: bulk load from a directory path on the server, as a job."""

from __future__ import annotations

import uuid  # noqa: TC003 - FastAPI resolves route annotations at runtime
from typing import TYPE_CHECKING

from fastapi import HTTPException, Query, status

from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves route annotations at runtime
from src.core.disk_space import DISK_GUARD
from src.routers.ingest._router import router
from src.schemas.ingest import (
    DirectoryIngestAccepted,
    DirectoryIngestResponse,
    DirectoryIngestStatus,
)
from src.services.jobs import status as job_status
from src.services.jobs.service import cancel_job, rejection, start_job
from src.services.visual_search.directory_ingest import (
    DirectoryIngestError,
    DirectoryIngestOptions,
    check_directory,
)


if TYPE_CHECKING:
    from src.schemas.jobs import JobView


KIND = 'directory_ingest'


@router.post(
    '/directory',
    dependencies=[DISK_GUARD],
    response_model=DirectoryIngestAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {'description': 'Queued; read the job at status_url'},
        400: {'description': 'Invalid input or directory not found'},
        503: {'description': 'The job queue is unavailable'},
    },
    summary='Bulk ingest from directory',
    description="""
Bulk ingest all images from a server-side directory path, as a background job.

**Supported formats:** JPEG, PNG, WebP, BMP, TIFF

**Use case:** Initial library ingestion, server-side batch processing.

**Note:** This endpoint reads files from the server filesystem.
Ensure the directory path is accessible from within the container.

**Processing:**
- Answers 202 with the job id; read `GET /ingest/directory/{job_id}` for progress and the result
- Files are processed in batches of up to 64
- Partial failures are allowed - check the result for details
- A re-run of the same directory stores nothing twice
""",
)
async def ingest_directory(
    session: DbSessionDep,
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
        description='Run face detection and embedding extraction. '
        'Ignored (always off) when OP_FACES_ENABLED=false.',
    ),
    enable_clip: bool = Query(
        True,
        description='Run MobileCLIP global image embedding',
    ),
    defer_heavy_ops: bool = Query(
        False,
        description='Defer heavy operations for faster bulk ingestion',
    ),
) -> DirectoryIngestAccepted:
    """Queue the ingest of a server-side directory and answer with its job."""
    try:
        check_directory(directory)
    except DirectoryIngestError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    options = DirectoryIngestOptions(
        directory=directory,
        recursive=recursive,
        max_images=max_images,
        batch_size=batch_size,
        skip_duplicates=skip_duplicates,
        detect_near_duplicates=detect_near_duplicates,
        near_duplicate_threshold=near_duplicate_threshold,
        enable_ocr=enable_ocr,
        enable_detection=enable_detection,
        enable_faces=enable_faces,
        enable_clip=enable_clip,
        defer_heavy_ops=defer_heavy_ops,
    )
    started = await start_job(session, KIND, None, params=options.to_params())
    job_id = str(started.job.id)
    return DirectoryIngestAccepted(
        job_id=job_id, status='queued', status_url=f'/ingest/directory/{job_id}'
    )


async def _directory_job(session: DbSessionDep, job_id: uuid.UUID) -> JobView:
    view = await job_status.get_job(session, job_id, project=None)
    if view.kind != KIND:
        raise rejection(404, 'job_not_found', f'no directory ingest {job_id}')
    return view


@router.get('/directory/{job_id}', response_model=DirectoryIngestStatus)
async def directory_ingest_status(
    job_id: uuid.UUID, session: DbSessionDep
) -> DirectoryIngestStatus:
    """The job of one directory ingest: status, progress and, once it finished, the result."""
    view = await _directory_job(session, job_id)
    result = None
    if view.result and 'status' in view.result:
        result = DirectoryIngestResponse.model_validate(view.result)
    return DirectoryIngestStatus(
        job=view, directory=(view.params or {}).get('directory'), result=result
    )


@router.post('/directory/{job_id}/cancel', response_model=DirectoryIngestStatus)
async def cancel_directory_ingest(
    job_id: uuid.UUID, session: DbSessionDep
) -> DirectoryIngestStatus:
    """Stop after the batch in flight; images already indexed stay indexed."""
    await _directory_job(session, job_id)
    await cancel_job(session, job_id, None, requested_by='api')
    return await directory_ingest_status(job_id, session)
