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

import os
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.routers.curation._common import OpenSearchDep, _ensure_indexes, router


def _scores_enabled() -> bool:
    return os.environ.get('OP_SCORES_ENABLED', '').strip().lower() in {'1', 'true', 'yes', 'on'}


class ScoresComputeRequest(BaseModel):
    scorers: list[str] | None = Field(
        default=None,
        description='Scorer ids to run (see GET /curation/methods). None = every enabled scorer.',
    )


@router.post('/scores/compute')
async def scores_compute(
    payload: ScoresComputeRequest,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    if not _scores_enabled():
        raise HTTPException(
            status_code=400,
            detail='curation scoring is disabled (set OP_SCORES_ENABLED=1 to enable)',
        )
    await _ensure_indexes(opensearch)

    from src.services.curation.item_scores import available_scorers, job as scores_job

    valid = set(available_scorers())
    requested = payload.scorers if payload.scorers else sorted(valid)
    unknown = [s for s in requested if s not in valid]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f'unknown scorer(s): {unknown}; valid: {sorted(valid)}',
        )

    try:
        state = scores_job.start_job(opensearch, requested)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return state


@router.get('/scores/status')
async def scores_status() -> dict[str, Any]:
    from src.services.curation.item_scores import job as scores_job

    return scores_job.get_state()


@router.post('/scores/cancel')
async def scores_cancel() -> dict[str, Any]:
    from src.services.curation.item_scores import job as scores_job

    cancelled = scores_job.cancel_job()
    return {'cancelled': cancelled, **scores_job.get_state()}


@router.get('/scores/coverage')
async def scores_coverage(opensearch: OpenSearchDep) -> dict[str, Any]:
    await _ensure_indexes(opensearch)
    from src.services.curation.item_scores import job as scores_job

    coverage = await scores_job.compute_coverage(opensearch)
    return {'coverage': coverage}
