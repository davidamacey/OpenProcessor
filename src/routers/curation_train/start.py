"""``POST /train/start`` and ``POST /train/start_campaign``."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from src.core.logging import get_logger
from src.routers.curation._common import OpenSearchDep  # noqa: TC001 - used at runtime by FastAPI
from src.routers.curation_train.preflight import (
    refuse_empty_val_split,
    refuse_unknown_augmentation_preset,
    run_preflight,
)
from src.services.training import jobs as train_jobs
from src.services.training.gpu_arbiter import GpuArbiterStopFailedError
from src.services.training.job_models import TrainCampaignSpec, TrainJobSpec
from src.services.training.preflight_checks import (
    PreflightReport,  # noqa: TC001 - pydantic field type
)
from src.services.training.profiles import PROFILES_YOLO26


logger = get_logger(__name__)

router = APIRouter()


class StartTrainResponse(BaseModel):
    job_id: str
    preflight: PreflightReport


@router.post(
    '/start',
    response_model=StartTrainResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_train(
    spec: TrainJobSpec,
    opensearch: OpenSearchDep,
    force: Annotated[bool, Query(description='Bypass blocking preflight checks')] = False,
) -> StartTrainResponse:
    """Validate, run preflight, and write ``job.json``.

    Returns 422 with the full preflight report if any check is blocking
    and ``force=False``; 422 for an unknown augmentation preset and 422
    ``empty_val_split`` (empty validation split) even with ``force``. The trainer picks up the file
    out-of-band.
    """
    refuse_unknown_augmentation_preset(spec.augmentation)
    report = await run_preflight(spec, opensearch)
    refuse_empty_val_split(report)
    # Active run gets 409 specifically (precedes the generic 422). Without
    # ``force``, active-run is non-overridable: the trainer only handles
    # one run at a time.
    active_check = next(
        (c for c in report.checks if c.name == 'active_run' and c.severity == 'block'),
        None,
    )
    if active_check is not None and not force:
        raise HTTPException(
            status_code=409,
            detail={'message': active_check.message, 'preflight': report.model_dump()},
        )
    if report.blocked and not force:
        raise HTTPException(
            status_code=422,
            detail={'message': 'preflight blocked', 'preflight': report.model_dump()},
        )
    # GPU arbiter — pause the VLM worker (single-GPU) or stop the container
    # (dual-GPU) BEFORE the trainer picks the job up, and BEFORE job.json
    # is written. Fails closed -- claim_gpus_for_training raises
    # GpuArbiterStopFailedError when a claim needs to stop a configured
    # GPU-resident container and can't (docker SDK/socket unavailable, or
    # the stop itself failed). Starting anyway would run training right
    # next to that service on the same GPU, so refuse with 409 and never
    # reach write_job.
    from src.services.training.gpu_arbiter import claim_gpus_for_training

    try:
        await claim_gpus_for_training(spec.cuda_visible_devices)
    except GpuArbiterStopFailedError as exc:
        logger.warning('gpu_arbiter_claim_failed', error=str(exc))
        raise HTTPException(
            status_code=409,
            detail={
                'message': (
                    'cannot claim the requested GPU(s): a configured GPU-resident '
                    f'container could not be stopped ({exc})'
                ),
            },
        ) from exc
    try:
        job_id = await train_jobs.write_job(spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StartTrainResponse(job_id=job_id, preflight=report)


class StartCampaignResponse(BaseModel):
    campaign_id: str
    job_ids: list[str]


@router.post(
    '/start_campaign',
    response_model=StartCampaignResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_campaign(
    campaign: TrainCampaignSpec,
    opensearch: OpenSearchDep,
    force: Annotated[bool, Query(description='Bypass blocking preflight checks')] = False,
) -> StartCampaignResponse:
    """Submit a multi-size training campaign.

    Preflight runs once on a synthetic spec built from the first run; the
    rest of the runs share the same dataset / class set so a single
    preflight covers them all.
    """
    if not campaign.runs:
        raise HTTPException(status_code=400, detail='campaign requires at least one run')
    refuse_unknown_augmentation_preset(campaign.augmentation)

    first = campaign.runs[0]
    probe_spec = TrainJobSpec(
        dataset_export_dir=campaign.dataset_export_dir,
        include_classes=campaign.include_classes,
        single_cls=campaign.single_cls,
        cuda_visible_devices=campaign.cuda_visible_devices,
        model_size=first.model_size or 'm',
        profile=first.profile if first.profile in PROFILES_YOLO26 else 'custom',  # type: ignore[arg-type]
        hyperparameters=first.hyperparameters,
        augmentation=campaign.augmentation,
    )
    report = await run_preflight(probe_spec, opensearch)
    refuse_empty_val_split(report)
    if report.blocked and not force:
        raise HTTPException(
            status_code=422,
            detail={'message': 'preflight blocked', 'preflight': report.model_dump()},
        )

    # GPU arbiter — claim once for the entire campaign. The reconcile loop
    # in src/main.py releases when no run is left in a non-terminal state.
    # Fails closed -- see the matching comment in start_train above.
    from src.services.training.gpu_arbiter import claim_gpus_for_training

    try:
        await claim_gpus_for_training(campaign.cuda_visible_devices)
    except GpuArbiterStopFailedError as exc:
        logger.warning('gpu_arbiter_claim_failed', error=str(exc))
        raise HTTPException(
            status_code=409,
            detail={
                'message': (
                    'cannot claim the requested GPU(s): a configured GPU-resident '
                    f'container could not be stopped ({exc})'
                ),
            },
        ) from exc
    try:
        campaign_id, job_ids = await train_jobs.write_campaign(campaign)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StartCampaignResponse(campaign_id=campaign_id, job_ids=job_ids)
