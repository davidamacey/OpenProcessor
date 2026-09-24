"""Tests for ``POST /curation/test_holdout/freeze`` (audit remediation Phase 2,
P0-1) — see ``docs/design/audit-remediation-plan-2026-09.md`` "## Phase 2".

Before this phase the endpoint was completely broken end to end:

1. Its cohort filter referenced ``label_source`` values
   (``v6_original_label`` / ``hdd_user_label``) that nothing in the repo
   ever writes — 0 matches, always.
2. Its composite aggregation grouped on the bare ``hdd_source`` field,
   which is ``text`` + ``.keyword`` on the live index — OpenSearch throws
   ``illegal_argument_exception`` on that (surfaces as a 503), and even if
   it hadn't, the single ``size: 1000`` composite page silently dropped any
   strata beyond the first page.
3. A zero-row selection returned ``200`` with ``sha256('')`` instead of
   erroring — a freeze that freezes nothing was reported as a success.
4. ``test_holdout_sha`` was computed and returned but never persisted
   anywhere durable, so a bad freeze couldn't be diagnosed or reverted.
5. A second, more-correct (SHA1-deterministic, min-5-per-class-floor)
   implementation lived in ``an offline promotion script``
   and was never wired into this endpoint.

Every test below is written against the fixed behaviour; see the phase
report for the fail-then-pass evidence captured by temporarily reverting
``src/routers/curation/review.py``, ``src/routers/curation/_common.py``,
and ``src/services/curation/holdout.py`` to their pre-Phase-2 state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.services.curation import holdout as test_holdout_module
from src.services.curation.holdout import compute_holdout_sha, select_test_holdout


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_opensearch() -> AsyncMock:
    """Mirrors the ``fake_opensearch`` fixture in ``tests/curation/test_review_router.py``
    — an AsyncMock standing in for AsyncOpenSearch used by the router."""
    fake = AsyncMock()
    fake.indices = AsyncMock()
    fake.indices.exists = AsyncMock(return_value=True)
    fake.indices.create = AsyncMock(return_value={'acknowledged': True})
    fake.indices.refresh = AsyncMock(return_value={'_shards': {}})
    fake.indices.put_mapping = AsyncMock(return_value={'acknowledged': True})
    fake.search = AsyncMock(return_value={'hits': {'hits': [], 'total': {'value': 0}}})
    fake.count = AsyncMock(return_value={'count': 0})
    fake.bulk = AsyncMock(return_value={'errors': False, 'items': []})
    return fake


@pytest.fixture
def app_client(fake_opensearch: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.config.curation import CurationConfig
    from src.core.dependencies import get_opensearch
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    # Every successful freeze persists a record under the configured
    # curation state dir — point it at a per-test tmp dir so tests never
    # touch the real default state dir (which isn't writable on a bare
    # host anyway). test_freeze_persists_record overrides this itself to
    # make the location explicit in that test.
    monkeypatch.setattr(
        'src.services.curation.holdout.get_curation_config',
        lambda: CurationConfig(state_dir=tmp_path),
    )

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[get_opensearch] = lambda: fake_opensearch
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch

    with TestClient(app) as client:
        client.fake_os = fake_opensearch  # type: ignore[attr-defined]
        yield client


def _strata_response(
    buckets: list[dict[str, Any]], after_key: dict[str, Any] | None = None
) -> dict[str, Any]:
    strata: dict[str, Any] = {'buckets': buckets}
    if after_key is not None:
        strata['after_key'] = after_key
    return {
        'hits': {'hits': [], 'total': {'value': sum(b.get('doc_count', 0) for b in buckets)}},
        'aggregations': {'strata': strata},
    }


def _bucket(class_id: int, hdd_source: str, doc_count: int) -> dict[str, Any]:
    """A composite-agg bucket -- key + doc_count only. The endpoint no
    longer carries a ``top_hits`` sample per bucket (that 400s past
    OpenSearch's default ``index.max_inner_result_window`` on any real
    stratum over 100 crops); it real-scans each stratum separately via
    :func:`_make_search_dispatcher`'s per-stratum branch below."""
    return {'key': {'class_id': class_id, 'hdd_source': hdd_source}, 'doc_count': doc_count}


def _crop_ids(prefix: str, n: int) -> list[str]:
    return [f'{prefix}_{i:03d}' for i in range(n)]


def _make_search_dispatcher(
    strata_pages: list[tuple[list[dict[str, Any]], dict[str, Any] | None]],
    crop_ids_by_stratum: dict[tuple[int, str], list[str]],
) -> Any:
    """Route a mocked ``opensearch.search`` call to either the composite-agg
    strata enumeration or a per-stratum ``search_after`` scan, based on the
    request body shape -- mirrors the two kinds of query the fixed endpoint
    actually issues (:func:`_fetch_holdout_cohort_strata` +
    :func:`_scan_stratum_crop_ids` in ``src/routers/curation/review.py``).

    ``strata_pages`` is consumed in order, one entry per composite-agg call
    (however many are made); each per-stratum scan returns its full
    crop_id list on the first page and an empty page after that (every
    test fixture here fits in one scan page).
    """
    state = {'page': 0}

    async def _dispatch(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        body = kwargs['body']
        aggs = body.get('aggs') or {}
        if 'strata' in aggs:
            idx = state['page']
            state['page'] += 1
            if idx >= len(strata_pages):
                return _strata_response([])
            buckets, after_key = strata_pages[idx]
            return _strata_response(buckets, after_key=after_key)

        must = body['query']['bool']['must']
        class_id = next(
            m['term']['class_id'] for m in must if 'term' in m and 'class_id' in m['term']
        )
        hdd_source = next(
            m['term']['hdd_source'] for m in must if 'term' in m and 'hdd_source' in m['term']
        )
        if body.get('search_after') is not None:
            return {'hits': {'hits': []}}
        ids = crop_ids_by_stratum.get((class_id, hdd_source), [])
        return {
            'hits': {'hits': [{'_source': {'crop_id': cid}, 'sort': [cid, cid]} for cid in ids]}
        }

    return _dispatch


# =============================================================================
# 1. Cohort filter
# =============================================================================


def test_freeze_selects_human_validated_cohort(app_client: Any, fake_opensearch: AsyncMock) -> None:
    """The composite-agg query must select ``class_validated=true AND
    class_source='human'`` — not the old ``label_source in
    ['v6_original_label', 'hdd_user_label']`` clause, which matches 0 rows
    on the live index.

    Before: builds the ``label_source`` terms query -> this assertion on
    the actual request body fails.
    """
    buckets = [_bucket(3, 'hdd:demo_hdd01', 6)]
    fake_opensearch.search = AsyncMock(
        side_effect=_make_search_dispatcher(
            [(buckets, None)], {(3, 'hdd:demo_hdd01'): _crop_ids('a', 6)}
        )
    )

    r = app_client.post('/curation/test_holdout/freeze', json={'percent': 20})
    assert r.status_code == 200, r.text

    # The composite-agg call is always issued before any per-stratum scan.
    call = fake_opensearch.search.call_args_list[0]
    body = call.kwargs['body']
    assert body['query'] == {
        'bool': {
            'must': [
                {'term': {'class_validated': True}},
                {'term': {'class_source': 'human'}},
            ]
        }
    }


# =============================================================================
# 2. Composite agg on hdd_source
# =============================================================================


def test_freeze_aggregates_on_keyword_subfield(app_client: Any, fake_opensearch: AsyncMock) -> None:
    """The composite agg source for ``hdd_source`` must use the bare
    ``hdd_source`` field name — it is mapped ``keyword`` directly on the
    live index (no ``.keyword`` sub-field exists), so appending
    ``.keyword`` would 400 against a real index (``illegal_argument_exception``,
    the same class of bug review_queries.py's model_disagreements script
    guard exists for).
    """
    buckets = [_bucket(3, 'hdd:demo_hdd01', 6)]
    fake_opensearch.search = AsyncMock(
        side_effect=_make_search_dispatcher(
            [(buckets, None)], {(3, 'hdd:demo_hdd01'): _crop_ids('a', 6)}
        )
    )

    r = app_client.post('/curation/test_holdout/freeze', json={'percent': 20})
    assert r.status_code == 200, r.text

    call = fake_opensearch.search.call_args_list[0]
    body = call.kwargs['body']
    sources = body['aggs']['strata']['composite']['sources']
    hdd_source_field = next(s['hdd_source']['terms']['field'] for s in sources if 'hdd_source' in s)
    assert hdd_source_field == 'hdd_source'


# =============================================================================
# 3. after_key pagination
# =============================================================================


def test_freeze_paginates_beyond_one_composite_page(
    app_client: Any, fake_opensearch: AsyncMock
) -> None:
    """A cohort with more distinct (class_id, hdd_source) strata than fit
    on one composite page must not be silently truncated to page 1 — the
    endpoint must follow ``after_key`` until it's absent, and every
    stratum's crops must make it into the final selection.

    Before: a single ``size: 1000`` composite page with no ``after_key``
    follow-up -> only the first page's class ever appears in the response,
    the second page's class is silently dropped.
    """
    after_key_1 = {'class_id': 1, 'hdd_source': 'hdd:demo_hdd01'}
    strata_pages = [
        ([_bucket(1, 'hdd:demo_hdd01', 6)], after_key_1),
        ([_bucket(2, 'hdd:demo_hdd01', 6)], None),
    ]
    crop_ids_by_stratum = {
        (1, 'hdd:demo_hdd01'): _crop_ids('p1', 6),
        (2, 'hdd:demo_hdd01'): _crop_ids('p2', 6),
    }
    fake_opensearch.search = AsyncMock(
        side_effect=_make_search_dispatcher(strata_pages, crop_ids_by_stratum)
    )

    r = app_client.post('/curation/test_holdout/freeze', json={'percent': 20})
    assert r.status_code == 200, r.text
    body = r.json()

    # Both pages' classes must be represented -- proves after_key was
    # followed and the second page wasn't dropped.
    assert '1' in body['per_class_counts']
    assert '2' in body['per_class_counts']

    composite_calls = [
        c
        for c in fake_opensearch.search.call_args_list
        if 'strata' in (c.kwargs['body'].get('aggs') or {})
    ]
    assert len(composite_calls) == 2, 'expected exactly 2 composite-agg pages to be fetched'

    # The second composite-agg call must carry the first call's after_key.
    assert composite_calls[1].kwargs['body']['aggs']['strata']['composite']['after'] == after_key_1


# =============================================================================
# 4. Zero-row freeze must error
# =============================================================================


def test_freeze_zero_rows_raises(app_client: Any, fake_opensearch: AsyncMock) -> None:
    """A cohort that resolves to zero crops must 422, not 200 with
    ``sha256('')``. A freeze that freezes nothing is never a success.

    Before: 200 with ``n_frozen=0`` and a bulk write of zero updates.
    """
    fake_opensearch.search = AsyncMock(side_effect=_make_search_dispatcher([([], None)], {}))

    r = app_client.post('/curation/test_holdout/freeze', json={'percent': 20})
    assert r.status_code == 422, r.text
    assert 'zero' in r.text.lower()
    fake_opensearch.bulk.assert_not_called()


# =============================================================================
# 5. Persistence
# =============================================================================


def test_freeze_persists_record(
    app_client: Any, fake_opensearch: AsyncMock, tmp_path: Path
) -> None:
    """A successful freeze must write a durable artifact under the
    configured curation state dir's ``test_holdout/`` subdirectory (the
    ``app_client`` fixture already points that at ``tmp_path``) -- a
    timestamped snapshot plus ``current.json`` -- whose sha matches the
    response's ``test_holdout_sha``. Mirrors the ``class_registry.<ISO>.json``
    snapshot convention.

    Before: ``test_holdout_sha`` is computed and returned but nothing is
    ever written to disk -- no directory, no files.
    """
    buckets = [_bucket(3, 'hdd:demo_hdd01', 6)]
    fake_opensearch.search = AsyncMock(
        side_effect=_make_search_dispatcher(
            [(buckets, None)], {(3, 'hdd:demo_hdd01'): _crop_ids('a', 6)}
        )
    )

    r = app_client.post('/curation/test_holdout/freeze', json={'percent': 20})
    assert r.status_code == 200, r.text
    body = r.json()

    state_dir = tmp_path / 'test_holdout'
    current_path = state_dir / 'current.json'
    assert current_path.exists(), f'no persisted freeze record under {state_dir}'
    record = json.loads(current_path.read_text())
    assert record['test_holdout_sha'] == body['test_holdout_sha']
    assert record['n_frozen'] == body['n_frozen']
    assert sorted(record['crop_ids']) == record['crop_ids']

    # A timestamped snapshot must exist alongside current.json.
    snapshots = [p for p in state_dir.iterdir() if p.name != 'current.json']
    assert len(snapshots) == 1, f'expected exactly one snapshot, found {snapshots}'
    assert json.loads(snapshots[0].read_text())['test_holdout_sha'] == body['test_holdout_sha']


# =============================================================================
# 6. Determinism
# =============================================================================


def test_freeze_is_deterministic(app_client: Any, fake_opensearch: AsyncMock) -> None:
    """Freezing the same cohort twice must select the identical crop-id
    set and sha -- no seed, no randomness. The plan doesn't call out an
    explicit pre-Phase-2 failure mode for this one (the old
    ``random.Random(payload.seed)`` with the request's default seed
    happens to be reproducible too, for a fixed seed) -- this test exists
    to pin the new algorithm's core guarantee: reproducibility without a
    seed to record, which is exactly why it was chosen (Appendix C
    Decision 1).
    """
    buckets = [_bucket(1, 'hdd:demo_hdd01', 9), _bucket(2, 'hdd:demo_hdd01', 3)]
    crop_ids_by_stratum = {
        (1, 'hdd:demo_hdd01'): _crop_ids('c1', 9),
        (2, 'hdd:demo_hdd01'): _crop_ids('c2', 3),
    }

    fake_opensearch.search = AsyncMock(
        side_effect=_make_search_dispatcher([(buckets, None)], crop_ids_by_stratum)
    )
    r1 = app_client.post('/curation/test_holdout/freeze', json={'percent': 20})
    assert r1.status_code == 200, r1.text
    body1 = r1.json()

    fake_opensearch.count = AsyncMock(return_value={'count': body1['n_frozen']})
    fake_opensearch.search = AsyncMock(
        side_effect=_make_search_dispatcher([(buckets, None)], crop_ids_by_stratum)
    )
    r2 = app_client.post('/curation/test_holdout/freeze?force=true', json={'percent': 20})
    assert r2.status_code == 200, r2.text
    body2 = r2.json()

    assert body1['test_holdout_sha'] == body2['test_holdout_sha']
    assert body1['per_class_counts'] == body2['per_class_counts']
    assert body1['n_frozen'] == body2['n_frozen']

    bulk_call_1 = fake_opensearch.bulk.call_args_list[0]
    bulk_call_2 = fake_opensearch.bulk.call_args_list[1]
    ids_1 = sorted(step['update']['_id'] for step in bulk_call_1.kwargs['body'] if 'update' in step)
    ids_2 = sorted(step['update']['_id'] for step in bulk_call_2.kwargs['body'] if 'update' in step)
    assert ids_1 == ids_2


# =============================================================================
# 7. Single implementation
# =============================================================================


def test_single_holdout_implementation() -> None:
    """There must be exactly one holdout-selection algorithm. Per Appendix
    C Decision 1, the SHA1-deterministic-per-class approach is
    canonicalized in ``src.services.curation.holdout.select_test_holdout``
    and every caller (the review router; an offline promotion script on a
    deployment that has one) must call it rather than reimplementing it.

    This branch has no offline promotion script to cross-check against
    (that script is deployment-specific tooling, not part of the generic
    curation package), so this test pins the one binding this branch does
    own: the review router's ``select_test_holdout`` is the exact same
    function object as the canonical module's, not a copy.
    """
    import src.routers.curation.review as review_module

    assert review_module.select_test_holdout is test_holdout_module.select_test_holdout


# =============================================================================
# Unit-level coverage of the shared algorithm itself
# =============================================================================


def test_select_test_holdout_min_five_floor_and_determinism() -> None:
    by_class = {1: _crop_ids('x', 3), 2: _crop_ids('y', 50)}
    chosen_a, per_class_a = select_test_holdout(by_class, fraction=0.1)
    chosen_b, per_class_b = select_test_holdout(by_class, fraction=0.1)

    assert chosen_a == chosen_b
    assert per_class_a == per_class_b
    # class 1 only has 3 candidates -- floor of 5 caps at the bucket size.
    assert per_class_a['1'] == 3
    # class 2 has 50 candidates at fraction 0.1 -> 5, at/above the floor.
    assert per_class_a['2'] == 5
    assert compute_holdout_sha(chosen_a) == compute_holdout_sha(list(reversed(chosen_a)))


# =============================================================================
# F-8 — class_id 0 must not be misbucketed as "unknown", and docs missing
# class_id/hdd_source must get an explicit stratum instead of being
# silently dropped from the composite agg.
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_cohort_strata_class_zero_is_not_treated_as_missing() -> None:
    """``int(key.get('class_id') or -1)`` collapses class_id == 0 into the
    "unknown" sentinel (-1), because ``0 or -1`` is ``-1`` in Python. A
    class-0 stratum must come back with class_id == 0."""
    from src.services.curation.holdout import fetch_cohort_strata

    fake = AsyncMock()

    async def _dispatch(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        body = kwargs['body']
        if 'strata' in (body.get('aggs') or {}):
            return _strata_response([_bucket(0, 'hdd:demo_hdd01', 6)])
        return {
            'hits': {
                'hits': [
                    {'_source': {'crop_id': cid}, 'sort': [cid, cid]} for cid in _crop_ids('z', 6)
                ]
            }
        }

    fake.search = AsyncMock(side_effect=_dispatch)

    strata = await fetch_cohort_strata(fake, 'test_items', {'match_all': {}})
    assert [b['class_id'] for b in strata] == [0]


@pytest.mark.asyncio
async def test_fetch_cohort_strata_missing_class_id_and_hdd_source_get_a_stratum() -> None:
    """A doc with no class_id/hdd_source must still surface as its own
    stratum (composite ``missing_bucket: true``), not vanish from the
    strata enumeration entirely."""
    from src.services.curation.holdout import (
        _MISSING_CLASS_ID_STRATUM,
        _MISSING_HOLDOUT_SOURCE_STRATUM,
        fetch_cohort_strata,
    )

    fake = AsyncMock()

    async def _dispatch(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        body = kwargs['body']
        if 'strata' in (body.get('aggs') or {}):
            return _strata_response(
                [{'key': {'class_id': None, 'hdd_source': None}, 'doc_count': 2}]
            )
        # Per-stratum scan for the missing-key bucket: assert it queries by
        # must_not exists rather than a literal term match on the sentinel.
        must = body['query']['bool']['must']
        assert {'bool': {'must_not': [{'exists': {'field': 'class_id'}}]}} in must
        assert {'bool': {'must_not': [{'exists': {'field': 'hdd_source'}}]}} in must
        return {
            'hits': {
                'hits': [
                    {'_source': {'crop_id': cid}, 'sort': [cid, cid]} for cid in _crop_ids('m', 2)
                ]
            }
        }

    fake.search = AsyncMock(side_effect=_dispatch)

    strata = await fetch_cohort_strata(fake, 'test_items', {'match_all': {}})
    assert len(strata) == 1
    assert strata[0]['class_id'] == _MISSING_CLASS_ID_STRATUM
    assert strata[0]['hdd_source'] == _MISSING_HOLDOUT_SOURCE_STRATUM
    assert strata[0]['crop_ids'] == _crop_ids('m', 2)
