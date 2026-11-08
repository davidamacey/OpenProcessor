"""The dataset upload as rows and a job: ``dataset_uploads`` beside ``dataset_upload_extract``.

The upload route streams the archive into the storage seam (``projects/<slug>/uploads/<id>.archive``)
and queues the extraction; the worker extracts it content-addressed under the project's upload
root, deletes the received archive and records the outcome on the row. No extraction runs in the
request (CC-5).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import update

from src.clients.storage.roots import build_rooted_store
from src.config.curation_accessors import get_curation_config
from src.config.project_context import current_project, current_project_id
from src.config.storage import storage_root_paths
from src.db.models.datasets import DatasetUpload
from src.db.session import project_tx
from src.domain.storage_keys import make_key
from src.services.curation.dataset_import import limits
from src.services.curation.dataset_import.upload import (
    ArchiveInvalidError,
    extract_received,
    receive_to_file,
)
from src.services.jobs import service as jobs


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repos.jobs import JobRow
    from src.tasks.runtime.context import JobContext

FILENAME_MAX = 255


def archive_key(slug: str, upload_id: uuid.UUID) -> str:
    return make_key('projects', slug, 'uploads', f'{upload_id}.archive')


def archive_path(key: str) -> Path:
    return build_rooted_store(storage_root_paths()).local_path(key)


async def _set_status(
    session: AsyncSession, project: uuid.UUID, upload_id: uuid.UUID, status: str, **values: Any
) -> None:
    async with project_tx(session, project):
        await session.execute(
            update(DatasetUpload)
            .where(DatasetUpload.id == upload_id)
            .values(status=status, **values)
        )


async def receive_and_queue(
    session: AsyncSession, stream: AsyncIterator[bytes], *, filename: str
) -> tuple[uuid.UUID, JobRow]:
    """Stream the archive to storage and queue its extraction: ``(upload id, job)``.

    The ``dataset_uploads`` row goes ``receiving`` -> ``received`` in the transaction that inserts
    the job. A refused stream or a refused start marks the row ``failed`` and removes the blob.
    """
    project = current_project_id()
    upload_id = uuid.uuid4()
    key = archive_key(current_project().record.slug, upload_id)
    expires = dt.datetime.now(dt.UTC) + dt.timedelta(hours=limits.upload_ttl_hours())
    async with project_tx(session, project):
        session.add(
            DatasetUpload(
                id=upload_id,
                project_id=project,
                filename=filename[:FILENAME_MAX],
                size_bytes=0,
                storage_key=key,
                status='receiving',
                expires_at=expires,
            )
        )
    path = archive_path(key)
    try:
        size, digest = await receive_to_file(stream, path)

        async def received(rows: AsyncSession, job: JobRow) -> None:
            await rows.execute(
                update(DatasetUpload)
                .where(DatasetUpload.id == upload_id)
                .values(size_bytes=size, sha256=digest, status='received', extract_job_id=job.id)
            )

        started = await jobs.start_job(
            session,
            'dataset_upload_extract',
            project,
            params={'upload_id': str(upload_id)},
            with_rows=received,
        )
    except BaseException:
        path.unlink(missing_ok=True)
        await _set_status(session, project, upload_id, 'failed')
        raise
    return upload_id, started.job


async def run_extract(ctx: JobContext) -> dict[str, Any]:
    """The ``dataset_upload_extract`` job body (the project is bound by the task)."""
    upload_id = uuid.UUID(str(ctx.params['upload_id']))
    async with ctx.session() as session:
        row = await session.get(DatasetUpload, upload_id)
        if row is None or row.sha256 is None:
            raise ArchiveInvalidError(f'upload {upload_id} was never received')
        key, digest, size = row.storage_key, row.sha256, row.size_bytes
        row.status = 'extracting'
    await ctx.progress(0, size, 'extracting')
    path = archive_path(key)
    try:
        extracted = await asyncio.to_thread(
            extract_received,
            path,
            sha256=digest,
            size=size,
            upload_root=Path(get_curation_config().upload_root),
        )
    except ArchiveInvalidError:
        async with ctx.session() as session:
            await session.execute(
                update(DatasetUpload).where(DatasetUpload.id == upload_id).values(status='failed')
            )
        path.unlink(missing_ok=True)
        raise
    async with ctx.session() as session:
        await session.execute(
            update(DatasetUpload).where(DatasetUpload.id == upload_id).values(status='extracted')
        )
    path.unlink(missing_ok=True)
    return {
        'upload_id': str(upload_id),
        'dataset_path': str(extracted.dataset_path),
        'bytes': extracted.bytes,
        'files': extracted.files,
    }
