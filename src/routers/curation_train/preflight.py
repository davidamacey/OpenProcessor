"""``POST /train/preflight``; the checks themselves are in
:mod:`src.services.training.preflight_run`."""

from __future__ import annotations

from fastapi import APIRouter

from src.routers.curation._common import OpenSearchDep  # noqa: TC001 - used at runtime by FastAPI
from src.services.training.job_models import TrainJobSpec  # noqa: TC001 - FastAPI body type
from src.services.training.preflight_checks import PreflightReport
from src.services.training.preflight_run import run_preflight


router = APIRouter()


@router.post('/preflight', response_model=PreflightReport)
async def preflight(spec: TrainJobSpec, opensearch: OpenSearchDep) -> PreflightReport:
    """Run every preflight gate and return the bundled report.

    The frontend uses this for inline validation while the user is
    filling the form (no side effects). ``/start`` calls this internally.
    """
    return await run_preflight(spec, opensearch)
