"""Training campaigns: queue one run per entry, cancel every run of a campaign.

Split out of ``jobs.py``. A campaign is a set of ``train`` jobs sharing ``params.campaign_id``;
the trainer queue runs them one at a time in the order they were queued.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.config.project_context import current_project_id
from src.core.logging import get_logger
from src.db.repos import training_runs as runs_repo
from src.db.session import async_session_factory, project_tx
from src.domain import job_states as js
from src.services.jobs import service as job_service
from src.services.training import jobs
from src.services.training.job_models import TrainCampaignSpec, TrainJobSpec


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


logger = get_logger(__name__)

CAMPAIGN_ID_RE = re.compile(r'^[A-Za-z0-9_.\-:+T]{1,128}$')
_VALID_PROFILES = {'probe', 'nano', 'small', 'medium', 'large', 'xlarge', 'custom'}


def _validate_campaign_id(campaign_id: str) -> None:
    if not CAMPAIGN_ID_RE.match(campaign_id):
        msg = f'invalid campaign_id: {campaign_id!r}'
        raise ValueError(msg)


async def write_campaign(
    session: AsyncSession, campaign: TrainCampaignSpec
) -> tuple[str, list[str]]:
    """Queue one run per ``runs[]`` entry in order; returns ``(campaign_id, [job_id, ...])``.

    A run that fails to queue cancels the ones already queued, so a campaign is never left
    half-started.
    """
    spec = campaign.model_copy(deep=True)
    if not spec.campaign_id:
        spec.campaign_id = f'{datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")}_campaign'
    _validate_campaign_id(spec.campaign_id)

    job_ids: list[str] = []
    last_idx = len(spec.runs) - 1
    try:
        for idx, run in enumerate(spec.runs):
            # An unknown profile name falls through to ``custom`` so the trainer still gets a
            # hyperparameter-only spec.
            profile: str = run.profile if run.profile in _VALID_PROFILES else 'custom'
            job_spec = TrainJobSpec(
                campaign_id=spec.campaign_id,
                submitted_by=spec.submitted_by,
                model_family='yolo26',
                model_size=run.model_size or 'm',
                profile=profile,
                dataset_export_dir=spec.dataset_export_dir,
                include_classes=spec.include_classes,
                single_cls=spec.single_cls,
                cuda_visible_devices=spec.cuda_visible_devices,
                hyperparameters=run.hyperparameters,
                augmentation=spec.augmentation,
                stop_when=spec.stop_when,
                auto_promote_best=spec.auto_promote_best,
                is_last_in_campaign=(idx == last_idx),
            )
            job_ids.append(await jobs.start_run(session, job_spec, campaign_id=spec.campaign_id))
    except Exception:
        for queued in job_ids:
            await jobs.cancel_run(queued)
        raise

    logger.info(
        'curation_train_campaign_written', campaign_id=spec.campaign_id, n_runs=len(job_ids)
    )
    return spec.campaign_id, job_ids


async def cancel_campaign(campaign_id: str) -> int:
    """Cancel every run of ``campaign_id`` that has not finished; returns how many."""
    _validate_campaign_id(campaign_id)
    project = current_project_id()
    async with async_session_factory()() as session:
        async with project_tx(session, project):
            runs = await session.run_sync(lambda sync: runs_repo.in_campaign(sync, campaign_id))
        cancelled = 0
        for run in runs:
            if run.status in js.TERMINAL:
                continue
            await job_service.cancel_job(session, run.job_id, project, requested_by='api')
            cancelled += 1
    logger.info('curation_train_campaign_cancelled', campaign_id=campaign_id, n=cancelled)
    return cancelled
