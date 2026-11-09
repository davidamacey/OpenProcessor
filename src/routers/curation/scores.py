"""Curation-score overlay job control.

``POST /curation/scores/compute`` / ``GET /curation/scores/status`` / ``POST
/curation/scores/cancel`` / ``GET /curation/scores/coverage``. New module (not touching
``review.py`` / ``pipeline.py`` — both already at the 700-LOC
pre-commit ceiling). Side-effect import: registers ``@router`` handlers on
the shared ``_common.router``.

Every scorer this endpoint can run is gated by ``OP_SCORES_ENABLED``
(default off — nothing behaviorally changes for an operator who doesn't
touch it). Read-only endpoints (``status``, ``coverage``) work regardless,
so an operator can always see the current state even with the flag off.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.config.env import env_bool
from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers.curation._common import OpenSearchDep, router


def _scores_enabled() -> bool:
    return env_bool('OP_SCORES_ENABLED')


class ScoresComputeRequest(BaseModel):
    scorers: list[str] | None = Field(
        default=None,
        description='Scorer ids to run (see GET /curation/methods). None = every enabled scorer.',
    )


@router.post('/scores/compute', status_code=202)
async def scores_compute(payload: ScoresComputeRequest, session: DbSessionDep) -> dict[str, Any]:
    """Queue the scoring job (202 with its state; 409 `job_busy` while one is active)."""
    if not _scores_enabled():
        raise HTTPException(
            status_code=400,
            detail='curation scoring is disabled (set OP_SCORES_ENABLED=1 to enable)',
        )

    from src.services.curation.item_scores import available_scorers, job as scores_job

    valid = set(available_scorers())
    requested = payload.scorers if payload.scorers else sorted(valid)
    unknown = [s for s in requested if s not in valid]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f'unknown scorer(s): {unknown}; valid: {sorted(valid)}',
        )

    await scores_job.start(session, current_project_id(), requested)
    return await _state(session)


async def _state(session: Any) -> dict[str, Any]:
    from src.services.curation.item_scores import job as scores_job
    from src.services.curation.latest_job import latest_job

    return scores_job.state_dict(await latest_job(session, scores_job.KIND))


@router.get('/scores/status')
async def scores_status(session: DbSessionDep) -> dict[str, Any]:
    return await _state(session)


@router.post('/scores/cancel')
async def scores_cancel(session: DbSessionDep) -> dict[str, Any]:
    from src.services.curation.item_scores.job import KIND
    from src.services.curation.latest_job import cancel_latest

    cancelled = await cancel_latest(session, KIND)
    return {'cancelled': cancelled, **await _state(session)}


@router.get('/scores/coverage')
async def scores_coverage(opensearch: OpenSearchDep) -> dict[str, Any]:
    from src.services.curation.item_scores import job as scores_job

    coverage = await scores_job.compute_coverage(opensearch)
    return {'coverage': coverage}
