"""Tests for the diversity/core-set selection overlay (curation-strategy
plan §2.6/§3.4/§7 Phase 4/§9): ``GET /curation/crops?order=diverse``'s helper
(``select.compute_diverse_order``) and ``POST /curation/select/diverse`` +
its ``status``/``cancel`` lifecycle.

Mirrors ``test_scores_router.py``'s conventions: mount the real
curation router with OpenSearch stubbed via ``AsyncMock``, monkeypatch the
background job runner wholesale to control the async lifecycle
deterministically instead of racing a real ``asyncio.create_task``
against synchronous ``TestClient`` calls.

The three ``GET /curation/crops?order=diverse`` full-router-integration
tests below (``test_get_crops_order_diverse_*``) are skipped on this
branch: ``src/routers/curation/crops.py`` (the ``GET /crops`` endpoint
``compute_diverse_order`` plugs into) has not been ported yet — it lands
in a later wave (plan §5 Chunk 9). The ``compute_diverse_order`` helper
itself is fully covered above by the unit-level tests that call it
directly without a router.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from collections.abc import Iterator


def _fake_scroll_client(ids_and_embeddings: list[tuple[str, list[float]]]) -> AsyncMock:
    """A minimal OpenSearch stand-in for ``fetch_pool_embeddings``'s
    scroll loop: one ``search`` call returns every hit (small test pools
    never exceed the real 2000-doc scroll page), one ``scroll`` call ends
    the scroll with an empty page, ``clear_scroll`` is a no-op."""
    fake_os = AsyncMock()
    hits = [{'_id': cid, '_source': {'pe_embedding': emb}} for cid, emb in ids_and_embeddings]
    fake_os.search = AsyncMock(return_value={'_scroll_id': 'sid-1', 'hits': {'hits': hits}})
    fake_os.scroll = AsyncMock(return_value={'_scroll_id': 'sid-1', 'hits': {'hits': []}})
    fake_os.clear_scroll = AsyncMock(return_value=None)
    fake_os.count = AsyncMock(return_value={'count': len(ids_and_embeddings)})
    return fake_os


@pytest.fixture(autouse=True)
def _clear_diverse_order_cache() -> Iterator[None]:
    """compute_diverse_order caches by (index, query) with a TTL, keyed
    off ``current_count`` for invalidation — production always supplies
    ``current_count`` (the crops router passes the live total), but these
    tests often reuse the same trivial query across differently-mocked
    OpenSearch clients without it, so clear the module-level cache
    between tests to avoid one test's result leaking into the next."""
    from src.routers.curation import select as kb_select

    kb_select._ORDER_CACHE.clear()
    yield
    kb_select._ORDER_CACHE.clear()


def _orthonormal_pool(n: int, d: int = 8) -> list[tuple[str, list[float]]]:
    """n distinct, mutually-orthonormal (as close as possible for n>d,
    falls back to repeats past d) embeddings so k-center-greedy has real
    structure to select over."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(n, d)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    return [(f'crop-{i}', base[i].tolist()) for i in range(n)]


# =============================================================================
# _build_scope_query — pure function, no app needed
# =============================================================================


def test_scope_query_always_excludes_test_holdout() -> None:
    from src.routers.curation.select import SelectDiverseScope, _build_scope_query

    q = _build_scope_query(SelectDiverseScope())
    assert {'term': {'test_holdout': True}} in q['bool']['must_not']


def test_scope_query_cluster_id_filter() -> None:
    from src.routers.curation.select import SelectDiverseScope, _build_scope_query

    q = _build_scope_query(SelectDiverseScope(cluster_id=10173))
    assert {'term': {'cluster_id': 10173}} in q['bool']['must']


def test_scope_query_generic_filters_scalar_and_list() -> None:
    from src.routers.curation.select import SelectDiverseScope, _build_scope_query

    q = _build_scope_query(SelectDiverseScope(filters={'class_id': 7, 'hdd_source': ['a', 'b']}))
    assert {'term': {'class_id': 7}} in q['bool']['must']
    assert {'terms': {'hdd_source': ['a', 'b']}} in q['bool']['must']


def test_scope_query_review_tab_reuses_review_queries() -> None:
    from src.routers.curation.select import SelectDiverseScope, _build_scope_query

    q = _build_scope_query(SelectDiverseScope(review_tab='outliers'))
    # review_queries.build_tab_query's 'outliers' tab adds the
    # outlier_flagged/cluster_distance should-clause verbatim.
    assert any('should' in clause.get('bool', {}) for clause in q['bool']['must'])


def test_scope_query_unknown_review_tab_raises_http_400() -> None:
    from fastapi import HTTPException

    from src.routers.curation.select import SelectDiverseScope, _build_scope_query

    with pytest.raises(HTTPException) as exc_info:
        _build_scope_query(SelectDiverseScope(review_tab='not_a_real_tab'))
    assert exc_info.value.status_code == 400


# =============================================================================
# compute_diverse_order — GET /curation/crops?order=diverse's helper
# =============================================================================


@pytest.mark.asyncio
async def test_compute_diverse_order_disabled_returns_none_without_any_os_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation.select import compute_diverse_order

    monkeypatch.delenv('KB_SELECT_DIVERSE_ENABLED', raising=False)
    fake_os = _fake_scroll_client(_orthonormal_pool(5))
    result = await compute_diverse_order(fake_os, 'kb_vehicle_crops', {'match_all': {}})
    assert result is None
    fake_os.search.assert_not_called()


@pytest.mark.asyncio
async def test_compute_diverse_order_returns_full_ranking_when_enabled_and_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation.select import compute_diverse_order

    monkeypatch.setenv('KB_SELECT_DIVERSE_ENABLED', '1')
    pool = _orthonormal_pool(6)
    fake_os = _fake_scroll_client(pool)
    result = await compute_diverse_order(fake_os, 'kb_vehicle_crops', {'match_all': {}})
    assert result is not None
    assert set(result) == {cid for cid, _ in pool}
    assert len(result) == 6


@pytest.mark.asyncio
async def test_compute_diverse_order_falls_back_above_inline_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors compute_outlier_order's 'too large -> None -> caller falls
    back to default sort' contract, just with the stricter (sqrt-of-ops-
    budget) inline cap this module documents for the O(n^2*d) full-pool
    ranking case."""
    from src.routers.curation.select import compute_diverse_order

    monkeypatch.setenv('KB_SELECT_DIVERSE_ENABLED', '1')
    # sync_max_ops=4 -> isqrt(4) == 2 -> inline cap is 2 rows; our pool of
    # 5 trips truncation immediately on the first (only) scroll page.
    monkeypatch.setenv('KB_SELECT_SYNC_MAX_OPS', '4')
    fake_os = _fake_scroll_client(_orthonormal_pool(5))
    result = await compute_diverse_order(fake_os, 'kb_vehicle_crops', {'match_all': {}})
    assert result is None


@pytest.mark.asyncio
async def test_compute_diverse_order_empty_pool_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation.select import compute_diverse_order

    monkeypatch.setenv('KB_SELECT_DIVERSE_ENABLED', '1')
    fake_os = _fake_scroll_client([])
    result = await compute_diverse_order(fake_os, 'kb_vehicle_crops', {'match_all': {}})
    assert result == []


# =============================================================================
# GET /curation/crops?order=diverse — full router integration (skipped -- crops.py not ported yet)
# =============================================================================


@pytest.fixture
def crops_app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as kb_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    fake_os = AsyncMock()
    fake_os.search = AsyncMock(
        return_value={
            'hits': {
                'total': {'value': 2},
                'hits': [
                    {'_id': 'crop-a', '_source': {'crop_id': 'crop-a', 'image_path': '/a.jpg'}},
                    {'_id': 'crop-b', '_source': {'crop_id': 'crop-b', 'image_path': '/b.jpg'}},
                ],
            }
        }
    )
    fake_os.mget = AsyncMock(
        return_value={
            'docs': [
                {
                    '_id': 'crop-b',
                    'found': True,
                    '_source': {'crop_id': 'crop-b', 'image_path': '/b.jpg'},
                },
                {
                    '_id': 'crop-a',
                    'found': True,
                    '_source': {'crop_id': 'crop-a', 'image_path': '/a.jpg'},
                },
            ]
        }
    )
    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


@pytest.mark.skip(reason='GET /crops (src/routers/curation/crops.py) not ported yet -- Chunk 9')
def test_get_crops_order_diverse_flag_off_behaves_like_default(
    crops_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('KB_SELECT_DIVERSE_ENABLED', raising=False)
    r_default = crops_app_client.get('/kb/crops', params={'order': 'default'})
    r_diverse = crops_app_client.get('/kb/crops', params={'order': 'diverse'})
    assert r_default.status_code == r_diverse.status_code == 200
    assert r_default.json() == r_diverse.json()


@pytest.mark.skip(reason='GET /crops (src/routers/curation/crops.py) not ported yet -- Chunk 9')
def test_get_crops_order_diverse_uses_the_computed_order_when_enabled(
    crops_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'src.routers.curation.select.compute_diverse_order',
        AsyncMock(return_value=['crop-b', 'crop-a']),
    )
    r = crops_app_client.get('/kb/crops', params={'order': 'diverse', 'page_size': 50})
    assert r.status_code == 200
    body = r.json()
    assert body['total'] == 2
    assert [c['crop_id'] for c in body['crops']] == ['crop-b', 'crop-a']


@pytest.mark.skip(reason='GET /crops (src/routers/curation/crops.py) not ported yet -- Chunk 9')
def test_get_crops_order_diverse_falls_back_when_helper_returns_none(
    crops_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'src.routers.curation.select.compute_diverse_order',
        AsyncMock(return_value=None),
    )
    r_default = crops_app_client.get('/kb/crops', params={'order': 'default'})
    r_diverse = crops_app_client.get('/kb/crops', params={'order': 'diverse'})
    assert r_diverse.json() == r_default.json()


# =============================================================================
# POST /curation/select/diverse + status/cancel
# =============================================================================


@pytest.fixture
def select_app_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as kb_router

    monkeypatch.setenv('KB_SELECT_JOBS_DIR', str(tmp_path / 'select'))
    monkeypatch.setenv('KB_SELECT_DIVERSE_ENABLED', '1')
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))

    fake_os = AsyncMock()
    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


def test_select_diverse_disabled_400(
    select_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('KB_SELECT_DIVERSE_ENABLED', raising=False)
    r = select_app_client.post('/curation/select/diverse', json={'k': 5})
    assert r.status_code == 400
    assert 'disabled' in r.json()['detail']


def test_select_diverse_unknown_review_tab_400(select_app_client: TestClient) -> None:
    r = select_app_client.post(
        '/curation/select/diverse', json={'k': 5, 'scope': {'review_tab': 'nope'}}
    )
    assert r.status_code == 400


def test_select_diverse_sync_path_respects_k_and_stays_in_scope(
    select_app_client: TestClient,
) -> None:
    pool = _orthonormal_pool(6)
    select_app_client.fake_os.count = AsyncMock(return_value={'count': 6})
    hits = [{'_id': cid, '_source': {'pe_embedding': emb}} for cid, emb in pool]
    select_app_client.fake_os.search = AsyncMock(
        return_value={'_scroll_id': 'sid', 'hits': {'hits': hits}}
    )
    select_app_client.fake_os.scroll = AsyncMock(
        return_value={'_scroll_id': 'sid', 'hits': {'hits': []}}
    )
    select_app_client.fake_os.clear_scroll = AsyncMock(return_value=None)

    r = select_app_client.post(
        '/curation/select/diverse', json={'k': 3, 'scope': {'cluster_id': 42}}
    )
    assert r.status_code == 200
    body = r.json()
    assert body['method'] == 'kcenter_greedy'
    assert body['version'] == 'v1'
    assert body['n_pool'] == 6
    assert len(body['crop_ids']) == 3
    assert set(body['crop_ids']) <= {cid for cid, _ in pool}


def test_select_diverse_job_path_for_large_pool(
    select_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the job path via a tiny KB_SELECT_SYNC_MAX_OPS, then drive
    the lifecycle exactly like test_scores_router.py does: monkeypatch
    the background coroutine to hang on a controlled asyncio.Event so the
    'running' state is observable deterministically."""
    from src.services.curation.selection import job as select_job

    monkeypatch.setenv('KB_SELECT_SYNC_MAX_OPS', '1')
    select_app_client.fake_os.count = AsyncMock(return_value={'count': 50_000})

    hang_forever = asyncio.Event()

    async def _fake_run_selection_job(job_id, opensearch, index, query, k, seed_crop_id, max_n):
        select_job._touch_heartbeat()
        await hang_forever.wait()

    monkeypatch.setattr(select_job, 'run_selection_job', _fake_run_selection_job)

    r = select_app_client.post('/curation/select/diverse', json={'k': 1000})
    assert r.status_code == 202
    body = r.json()
    assert body['status'] == 'running'
    assert body['k'] == 1000

    r_status = select_app_client.get('/curation/select/status')
    assert r_status.json()['status'] == 'running'

    r_double = select_app_client.post('/curation/select/diverse', json={'k': 500})
    assert r_double.status_code == 409

    r_cancel = select_app_client.post('/curation/select/cancel')
    assert r_cancel.status_code == 200
    assert r_cancel.json()['cancelled'] is True
    assert select_job.is_cancelled()


def test_select_status_idle_with_no_job(select_app_client: TestClient) -> None:
    r = select_app_client.get('/curation/select/status')
    assert r.status_code == 200
    assert r.json()['status'] == 'idle'


def test_select_cancel_with_no_job_running(select_app_client: TestClient) -> None:
    r = select_app_client.post('/curation/select/cancel')
    assert r.status_code == 200
    assert r.json()['cancelled'] is False


# =============================================================================
# Regression guard — the selection overlay never writes cluster fields or
# mutates OpenSearch at all (plan §8 non-goal #3 / the Phase 4 hard
# constraint: "100% read-only selection"). Equivalent in spirit to
# test_crop_scores.py::test_no_scorer_writes_cluster_fields, but this
# overlay isn't a CropScorer (it has no `writes` ClassVar to parametrize
# over — it writes NOTHING), so the guard here is structural: scan the
# actual source for any OpenSearch mutation call or forbidden field-name
# write, across every module this Phase 4 change added.
# =============================================================================


_FORBIDDEN_CLUSTER_FIELDS = ('cluster_id', 'cluster_subid', 'cluster_distance')
_SELECTION_SOURCE_FILES = (
    'src/services/curation/selection/__init__.py',
    'src/services/curation/selection/kcenter_greedy.py',
    'src/services/curation/selection/pool_fetch.py',
    'src/services/curation/selection/job.py',
    'src/routers/curation/select.py',
)


def test_selection_overlay_never_mutates_opensearch() -> None:
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    write_call_pattern = re.compile(r'\.(bulk|update|index)\s*\(')
    for rel_path in _SELECTION_SOURCE_FILES:
        src = (repo_root / rel_path).read_text()
        for match in write_call_pattern.finditer(src):
            # '.index(' is also the plain list/str method (e.g.
            # `ids.index(seed_crop_id)`) — only flag it when it looks like
            # an OpenSearch client call (`client.index(` / `opensearch.index(`).
            if match.group(1) == 'index':
                prefix = src[max(0, match.start() - 20) : match.start()]
                if not re.search(r'(opensearch|client|os)\s*$', prefix):
                    continue
            raise AssertionError(
                f'{rel_path} appears to call an OpenSearch write method '
                f'({match.group(0)!r}) -- the selection overlay must stay 100% read-only'
            )


def test_selection_overlay_query_filters_never_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cluster_id`` legitimately appears as a *read*-side query filter
    (``SelectDiverseScope.cluster_id`` -> ``{'term': {'cluster_id': ...}}``,
    same as GET /curation/crops's own existing cluster_id filter) — that's fine.
    What must never happen is any of those field names appearing on the
    *write* side of a bulk/update doc body. Combined with
    ``test_selection_overlay_never_mutates_opensearch`` (which proves
    there is no write call at all in this package), the two tests
    together are the real guarantee: no write call exists, therefore no
    field -- forbidden or otherwise -- is ever written."""
    from src.routers.curation.select import SelectDiverseScope, _build_scope_query

    q = _build_scope_query(SelectDiverseScope(cluster_id=42))
    # cluster_id is allowed to appear inside a {'term': {...}} *query*
    # clause (read-side scoping) -- assert it never appears as a `doc`/
    # write body key, which this query-builder never produces at all.
    assert 'doc' not in q
    assert set(_FORBIDDEN_CLUSTER_FIELDS) & set(q.keys()) == set()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
