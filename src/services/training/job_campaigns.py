"""Training campaigns: write one job per run entry, cancel every job of a
campaign.

Split out of ``jobs.py``.
"""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.services.training.job_files import _slug, _validate_job_id
from src.services.training.job_models import TrainCampaignSpec, TrainJobSpec
from src.services.training.jobs import list_runs, write_cancel, write_job


logger = get_logger(__name__)


async def write_campaign(campaign: TrainCampaignSpec) -> tuple[str, list[str]]:
    """Materialize one ``job.json`` per ``runs[]`` entry in order.

    Returns ``(campaign_id, [job_id, ...])``. The trainer processes the
    files in the order they're written; we write smallest-to-largest so
    the user gets fast feedback first.
    """
    spec = campaign.model_copy(deep=True)
    if not spec.campaign_id:
        spec.campaign_id = f'{_slug()}_campaign'
    _validate_job_id(spec.campaign_id)

    valid_profiles = {'probe', 'nano', 'small', 'medium', 'large', 'xlarge', 'custom'}
    job_ids: list[str] = []
    last_idx = len(spec.runs) - 1
    for idx, run in enumerate(spec.runs):
        # Narrow ``profile`` to the literal set TrainJobSpec accepts;
        # fall through to ``custom`` for unknown names so the trainer can
        # still receive a hyperparameter override-only spec.
        profile_value: Any = run.profile if run.profile in valid_profiles else 'custom'
        job_spec = TrainJobSpec(
            job_id=f'{spec.campaign_id}_run{idx:02d}_{run.profile}',
            campaign_id=spec.campaign_id,
            submitted_by=spec.submitted_by,
            model_family='yolo26',
            model_size=run.model_size or 'm',
            profile=profile_value,
            dataset_export_dir=spec.dataset_export_dir,
            include_classes=spec.include_classes,
            single_cls=spec.single_cls,
            cuda_visible_devices=spec.cuda_visible_devices,
            hyperparameters=run.hyperparameters,
            augmentation=spec.augmentation,
            # Carry campaign-wide policy onto every per-job spec so the
            # trainer can act on them between runs (auto-skip +
            # auto-promote).
            stop_when=spec.stop_when,
            auto_promote_best=spec.auto_promote_best,
            is_last_in_campaign=(idx == last_idx),
        )
        jid = await write_job(job_spec)
        job_ids.append(jid)

    logger.info(
        'curation_train_campaign_written',
        campaign_id=spec.campaign_id,
        n_runs=len(job_ids),
    )
    return spec.campaign_id, job_ids


async def cancel_campaign(campaign_id: str) -> int:
    """Cancel every job belonging to ``campaign_id``.

    We walk every ``job.json`` in the directory, match on the embedded
    ``campaign_id``, and drop a cancel sentinel for non-terminal runs.
    Returns the count of sentinels written.
    """
    _validate_job_id(campaign_id)
    runs = await list_runs(limit=200)
    cancelled = 0
    for run in runs:
        if run.campaign_id != campaign_id:
            continue
        if run.state in ('finished', 'failed', 'cancelled', 'skipped'):
            continue
        await write_cancel(run.job_id)
        cancelled += 1
    logger.info('curation_train_campaign_cancelled', campaign_id=campaign_id, n=cancelled)
    return cancelled
