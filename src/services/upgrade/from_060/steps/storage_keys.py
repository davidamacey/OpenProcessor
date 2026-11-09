"""Step ``storage_keys``: rows written before the root layout was fixed get relative keys.

* ``bakeoff_runs.results_storage_key`` held the absolute ``.../bakeoff_jobs/out/<run>``
  directory; it becomes ``bakeoff_results/<slug>/<run>``, the same directory through the
  ``bakeoff_results`` storage root.
* ``models.artifact_storage_key`` of an uploaded model was ``pytorch_models/<file>``; it becomes
  ``models/pytorch_models/<file>``, which the store resolves to the separate pytorch_models mount.

No file moves: the keys name where the files already are. Idempotent (a key already in the new
form is left alone); the counts are logged and added to the load report.
"""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING

from sqlalchemy import select, update

from src.core.logging import get_logger
from src.db.models import BakeoffRun, Model, Project
from src.domain.storage_keys import make_key


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.upgrade.from_060.steps import StepContext

ORDER = 60
OLD_PYTORCH_PREFIX = 'pytorch_models/'
logger = get_logger(__name__)


async def rewrite_bakeoff_results(session: AsyncSession) -> int:
    rows = (
        await session.execute(
            select(BakeoffRun.job_id, Project.slug, BakeoffRun.results_storage_key)
            .join(Project, Project.id == BakeoffRun.project_id)
            .where(BakeoffRun.results_storage_key.is_not(None))
        )
    ).all()
    changed = 0
    for job_id, slug, old in rows:
        if old is None or old.startswith('bakeoff_results/'):
            continue
        run = posixpath.basename(old.rstrip('/'))
        await session.execute(
            update(BakeoffRun)
            .where(BakeoffRun.job_id == job_id)
            .values(results_storage_key=make_key('bakeoff_results', slug, run))
        )
        changed += 1
    return changed


async def rewrite_uploaded_model_keys(session: AsyncSession) -> int:
    rows = (
        await session.execute(
            select(Model.id, Model.artifact_storage_key).where(
                Model.artifact_storage_key.startswith(OLD_PYTORCH_PREFIX)
            )
        )
    ).all()
    for model_id, old in rows:
        await session.execute(
            update(Model).where(Model.id == model_id).values(artifact_storage_key=f'models/{old}')
        )
    return len(rows)


async def run(ctx: StepContext) -> None:
    bakeoffs = await rewrite_bakeoff_results(ctx.session)
    models = await rewrite_uploaded_model_keys(ctx.session)
    logger.info('storage_keys_rewritten', bakeoff_results=bakeoffs, uploaded_models=models)
    ctx.report.add('storage_keys_bakeoff_results', bakeoffs)
    ctx.report.add('storage_keys_uploaded_models', models)
