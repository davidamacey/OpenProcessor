"""``POST /train/start`` and ``POST /train/start_campaign``."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers.curation._common import OpenSearchDep  # noqa: TC001 - used at runtime by FastAPI
from src.services.training import job_campaigns, jobs as train_jobs
from src.services.training.job_models import TrainCampaignSpec, TrainJobSpec
from src.services.training.preflight_checks import PreflightReport
from src.services.training.preflight_run import (
    refuse_empty_val_split,
    refuse_unknown_augmentation_preset,
    run_preflight,
)
from src.services.training.profiles import PROFILES_YOLO26


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
    session: DbSessionDep,
    force: Annotated[bool, Query(description='Bypass blocking preflight checks')] = False,
) -> StartTrainResponse:
    """Validate, run preflight, and queue the run as a ``train`` job.

    Returns 422 with the full preflight report if any check is blocking
    and ``force=False``; 422 for an unknown augmentation preset and 422
    ``empty_val_split`` (empty validation split) even with ``force``. The trainer consumes the
    ``gpu_train`` queue.
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
    try:
        job_id = await train_jobs.start_run(session, spec)
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
    session: DbSessionDep,
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
        profile=first.profile if first.profile in PROFILES_YOLO26 else 'custom',
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

    try:
        campaign_id, job_ids = await job_campaigns.write_campaign(session, campaign)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StartCampaignResponse(campaign_id=campaign_id, job_ids=job_ids)
