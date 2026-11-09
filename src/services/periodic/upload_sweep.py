"""``op.periodic.upload_sweep``: remove extracted dataset uploads that nothing refers to.

An extracted archive lives in ``<upload_root>/datasets/<sha256[:16]>/`` (content addressed). It is
kept while its ``dataset_uploads`` row has not expired and for as long as any ``dataset_imports``
row names it: through the upload (``upload_id``) or through the ``source_sha`` of its scan, because
an import over a directory source copies its images into the same tree. Everything else older than
``OP_DATASET_UPLOAD_TTL_H`` goes, with the half-written ``.incoming`` leftovers of a crashed upload
(``sweep_uploads``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import text

from src.config.advisory_locks import UPLOAD_SWEEP
from src.config.curation_accessors import get_curation_config
from src.core.logging import get_logger
from src.db.session import async_session_factory, project_tx
from src.services.curation.dataset_import.upload import sweep_uploads
from src.services.periodic.runner import TickProject, TickReport, for_each_project


logger = get_logger(__name__)

_REFERENCED = text(
    "SELECT left(scan ->> 'source_sha', 16) FROM dataset_imports WHERE scan ? 'source_sha' "
    'UNION '
    'SELECT left(sha256, 16) FROM dataset_uploads WHERE sha256 IS NOT NULL AND ('
    'expires_at > now() OR id IN (SELECT upload_id FROM dataset_imports WHERE upload_id IS NOT NULL))'
)


async def tick() -> TickReport:
    async def one(project: TickProject) -> None:
        async with async_session_factory()() as session, project_tx(session, project.id):
            referenced = {row[0] for row in await session.execute(_REFERENCED) if row[0]}
        removed = await asyncio.to_thread(
            sweep_uploads, Path(get_curation_config().upload_root), referenced=referenced
        )
        if removed:
            logger.info('upload_sweep_removed', project=project.slug, uploads=removed)

    return await for_each_project(UPLOAD_SWEEP, one, task='op.periodic.upload_sweep')
