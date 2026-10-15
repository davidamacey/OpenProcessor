"""Curation router sub-module — frozen test holdout: ``POST /test_holdout/freeze`` and stats."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query

from src.routers.curation._common import OpenSearchDep, items_index, logger, router
from src.routers.curation._common_models import TestHoldoutFreezeRequest, TestHoldoutFreezeResponse
from src.services.curation.dataset_thresholds import MIN_TEST_CROPS_PER_CLASS
from src.services.curation.holdout import (
    MIN_TEST_PER_CLASS,
    build_cohort_query,
    compute_holdout_sha,
    fetch_cohort_strata,
    persist_freeze_record,
    select_test_holdout,
)
from src.services.curation.index_bootstrap import ensure_indexes
from src.utils.timeutil import now_iso


@router.post('/test_holdout/freeze', response_model=TestHoldoutFreezeResponse)
async def freeze_test_holdout(
    payload: TestHoldoutFreezeRequest,
    opensearch: OpenSearchDep,
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
    durable freeze record under ``OP_STATE_DIR/test_holdout/`` so a bad
    freeze can be diagnosed and reverted from the recorded crop-id list.
    """
    await ensure_indexes(opensearch)
    # Refuse re-run if an existing holdout exists.
    if not force:
        try:
            existing = await opensearch.count(
                index=items_index(),
                body={'query': {'term': {'test_holdout': True}}},
            )
            if (existing or {}).get('count', 0) > 0:
                raise HTTPException(
                    status_code=409,
                    detail='test holdout already exists; pass ?force=true to re-run',
                )
        except HTTPException:
            raise
        except Exception as exc:
            # Without the count the "a holdout already exists" guard cannot hold: refuse.
            logger.warning('curation_test_holdout_count_failed', error=str(exc))
            raise HTTPException(
                status_code=503, detail='could not check for an existing test holdout'
            ) from exc

    cohort_query = build_cohort_query()
    try:
        strata = await fetch_cohort_strata(opensearch, items_index(), cohort_query)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc

    crop_ids_by_class: dict[int, list[str]] = {}
    for bucket in strata:
        ids = bucket.get('crop_ids') or []
        if not ids:
            continue
        crop_ids_by_class.setdefault(bucket['class_id'], []).extend(ids)

    fraction = payload.percent / 100.0
    chosen_ids, per_class = select_test_holdout(crop_ids_by_class, fraction=fraction)

    if not chosen_ids:
        cohort_size = sum(len(v) for v in crop_ids_by_class.values())
        raise HTTPException(
            status_code=422,
            detail=(
                f'zero crops selected from a cohort of {cohort_size} across '
                f'{len(crop_ids_by_class)} classes; refusing to freeze nothing'
            ),
        )

    # Bulk-flag.
    bulk: list[dict[str, Any]] = []
    now = now_iso()
    for crop_id in chosen_ids:
        bulk.append({'update': {'_index': items_index(), '_id': crop_id}})
        bulk.append({'doc': {'test_holdout': True, 'updated_at': now}})
    try:
        await opensearch.bulk(body=bulk, refresh=True)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'bulk failed: {exc}') from exc

    digest = compute_holdout_sha(chosen_ids)

    persist_freeze_record(
        crop_ids=chosen_ids,
        sha=digest,
        cohort_spec={
            'query': cohort_query,
            'selection': 'sha1_per_class',
            'percent': payload.percent,
            'fraction': fraction,
            'force': force,
        },
        per_class_counts=per_class,
    )

    return TestHoldoutFreezeResponse(
        n_frozen=len(chosen_ids),
        n_classes_covered=len(per_class),
        test_holdout_sha=digest,
        per_class_counts=per_class,
        percent=payload.percent,
        min_per_class=MIN_TEST_PER_CLASS,
    )


@router.get('/test_holdout/stats')
async def test_holdout_stats(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Per-class test-set counts."""
    body = {
        'size': 0,
        'query': {'term': {'test_holdout': True}},
        'aggs': {'by_class': {'terms': {'field': 'class_id', 'size': 1000}}},
        'track_total_hits': True,
    }
    try:
        resp = await opensearch.search(index=items_index(), body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc
    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    buckets = (resp.get('aggregations') or {}).get('by_class', {}).get('buckets', [])
    return {
        'total': int(total),
        # deficient: below the per-class test minimum preflight warns on.
        'by_class': [
            {**b, 'deficient': int(b.get('doc_count', 0)) < MIN_TEST_CROPS_PER_CLASS}
            for b in buckets
        ],
        'min_test_per_class': MIN_TEST_CROPS_PER_CLASS,
    }
