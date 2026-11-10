"""Curation router sub-module — frozen test holdout: ``POST /test_holdout/freeze`` and stats."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import Depends, HTTPException, Query

from src.core.db_session import DbSessionDep
from src.routers._errors import server_error
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._holdout_models import TestHoldoutFreezeRequest, TestHoldoutFreezeResponse
from src.routers.curation.near_dup_gate import require_near_dup_freezable
from src.schemas.curation.holdout import HoldoutStats
from src.services.curation import holdout_freeze as svc


@router.post(
    '/test_holdout/freeze',
    response_model=TestHoldoutFreezeResponse,
    dependencies=[Depends(require_near_dup_freezable)],
)
async def freeze_test_holdout(
    payload: TestHoldoutFreezeRequest,
    opensearch: OpenSearchDep,
    session: DbSessionDep,
    force: bool = Query(False, description='Allow re-running freeze (overwrites existing).'),
) -> TestHoldoutFreezeResponse:
    """Deterministic per-class sample of human-validated crops to ``test_holdout=true``.

    Cohort: ``class_validated=true AND class_source='human'`` (Appendix C
    Decision 2). Selection: SHA1-of-``crop_id`` deterministic per-class
    sampling with a 5-crop floor via :func:`select_test_holdout` — the
    single canonical holdout algorithm shared with
    an offline promotion script (Appendix C
    Decision 1). No seed (a ``seed`` field is a ``422``): the same cohort
    always freezes the same set. The response names the method in
    ``selection`` (``sha1_per_class``).

    Refuses to run if a holdout already exists unless ``force=true``.
    Refuses (422) to freeze zero rows — a freeze that freezes nothing is
    never a success, and silently returning 200 with ``sha256('')`` hid
    that this endpoint has never actually frozen anything. Persists a
    durable freeze record (a ``holdout_freezes`` row and a snapshot of the crop-id
    list) so a bad freeze can be diagnosed and reverted. The row is written first
    inside the transaction that wraps the flagging: a database error refuses the
    freeze before any item changes, and a failed flagging rolls the row back.
    """
    try:
        result = await svc.freeze_test_holdout(
            opensearch, session, percent=payload.percent, force=force
        )
    except svc.HoldoutExistsError as exc:
        raise HTTPException(
            status_code=409, detail='test holdout already exists; pass ?force=true to re-run'
        ) from exc
    except svc.HoldoutCheckError as exc:
        raise HTTPException(
            status_code=503, detail='could not check for an existing test holdout'
        ) from exc
    except svc.HoldoutSearchError as exc:
        raise server_error(503, 'opensearch error', exc) from exc
    except svc.HoldoutFlagError as exc:
        raise server_error(503, 'bulk failed', exc) from exc
    except svc.EmptyHoldoutCohortError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TestHoldoutFreezeResponse(**asdict(result))


@router.get('/test_holdout/stats', response_model=HoldoutStats)
async def test_holdout_stats(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Per-class test-set counts."""
    try:
        return await svc.holdout_stats(opensearch)
    except Exception as exc:
        raise server_error(503, 'opensearch error', exc) from exc
