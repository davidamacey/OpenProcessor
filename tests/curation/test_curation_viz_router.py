"""Tests for the UMAP visualization-only projection overlay
(curation-strategy plan §2.7/§3.5/§7 Phase 5/§9): ``POST
/curation/viz/projection/rebuild`` + ``status``/``cancel`` lifecycle, and
``GET /curation/viz/projection``.

Mirrors the reference line's select/scores router test conventions:
mount the real curation router with OpenSearch stubbed, monkeypatch
the background job runner wholesale to control the async lifecycle
deterministically instead of racing a real ``asyncio.create_task``
against synchronous ``TestClient`` calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    from src.routers.curation import viz as viz_module
    from src.routers.curation._common import _raw_opensearch_dep, router as curation_router

    monkeypatch.setenv('OP_VIZ_JOBS_DIR', str(tmp_path / 'viz'))
    # viz.py binds `_ensure_indexes` into its own module namespace (`from
    # ..._common import _ensure_indexes`), so the patch target is the viz
    # module's own name, not `_common`'s — patching `_common._ensure_indexes`
    # would not affect viz.py's already-bound reference.
    monkeypatch.setattr(viz_module, '_ensure_indexes', AsyncMock(return_value=None))

    fake_os = AsyncMock()
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


# =============================================================================
# Feature-flag gating
# =============================================================================


def test_rebuild_disabled_400(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_VIZ_PROJECTION_ENABLED', raising=False)
    r = app_client.post('/curation/viz/projection/rebuild')
    assert r.status_code == 400
    assert 'disabled' in r.json()['detail']


def test_get_projection_disabled_400(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_VIZ_PROJECTION_ENABLED', raising=False)
    r = app_client.get('/curation/viz/projection')
    assert r.status_code == 400
    assert 'disabled' in r.json()['detail']


def test_status_and_cancel_are_not_gated_by_the_flag(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same convention as scores_status/select_status: read-only lifecycle
    endpoints work regardless of the feature flag, so an operator can
    always see current state."""
    monkeypatch.delenv('OP_VIZ_PROJECTION_ENABLED', raising=False)
    r_status = app_client.get('/curation/viz/projection/status')
    assert r_status.status_code == 200
    assert r_status.json()['status'] == 'idle'

    r_cancel = app_client.post('/curation/viz/projection/cancel')
    assert r_cancel.status_code == 200
    assert r_cancel.json()['cancelled'] is False


# =============================================================================
# POST /curation/viz/projection/rebuild — validation + job lifecycle
# =============================================================================


def test_rebuild_cluster_scope_without_cluster_id_400(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    r = app_client.post('/curation/viz/projection/rebuild', params={'scope': 'cluster'})
    assert r.status_code == 400
    assert 'cluster_id' in r.json()['detail']


def test_rebuild_unknown_scope_422(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    r = app_client.post('/curation/viz/projection/rebuild', params={'scope': 'bogus'})
    assert r.status_code == 422  # FastAPI Query(pattern=...) rejects before the handler runs


def test_rebuild_then_status_running_then_double_start_409_then_cancel(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import embedding_viz

    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')

    hang_forever = asyncio.Event()

    async def _fake_run_projection_job(job_id, opensearch, *, scope, cluster_id, max_n) -> None:
        embedding_viz._touch_heartbeat()
        await hang_forever.wait()

    monkeypatch.setattr(embedding_viz, 'run_projection_job', _fake_run_projection_job)

    r = app_client.post('/curation/viz/projection/rebuild', params={'scope': 'residual'})
    assert r.status_code == 202
    body = r.json()
    assert body['status'] == 'running'
    assert body['scope'] == 'residual'

    r_status = app_client.get('/curation/viz/projection/status')
    assert r_status.json()['status'] == 'running'

    r_double = app_client.post('/curation/viz/projection/rebuild', params={'scope': 'residual'})
    assert r_double.status_code == 409

    r_cancel = app_client.post('/curation/viz/projection/cancel')
    assert r_cancel.status_code == 200
    assert r_cancel.json()['cancelled'] is True
    assert embedding_viz.is_cancelled()

    hang_forever.set()


def test_rebuild_cluster_scope_passes_cluster_id_through(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import embedding_viz

    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')

    hang_forever = asyncio.Event()

    async def _fake_run_projection_job(job_id, opensearch, *, scope, cluster_id, max_n) -> None:
        embedding_viz._touch_heartbeat()
        await hang_forever.wait()

    monkeypatch.setattr(embedding_viz, 'run_projection_job', _fake_run_projection_job)

    r = app_client.post(
        '/curation/viz/projection/rebuild', params={'scope': 'cluster', 'cluster_id': 10173}
    )
    assert r.status_code == 202
    body = r.json()
    assert body['scope'] == 'cluster'
    assert body['cluster_id'] == 10173

    hang_forever.set()


# =============================================================================
# GET /curation/viz/projection — cached-coordinates-only serving
# =============================================================================


def test_get_projection_not_built(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    app_client.fake_os.get = AsyncMock(side_effect=Exception('not found'))

    r = app_client.get('/curation/viz/projection')
    assert r.status_code == 200
    assert r.json() == {'status': 'not_built'}
    app_client.fake_os.search.assert_not_called()


def test_get_projection_serves_cached_points(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    app_client.fake_os.get = AsyncMock(
        return_value={
            '_source': {
                'projection_version': 'umap_viz_v1',
                'fitted_at': '2026-09-11T00:00:00+00:00',
            }
        }
    )
    app_client.fake_os.search = AsyncMock(
        return_value={
            'hits': {
                'hits': [
                    {
                        '_id': 'crop-a',
                        '_source': {
                            'viz_x': 0.5,
                            'viz_y': -0.5,
                            'cluster_id': 42,
                            'class_name': 'sedan',
                            'class_source': 'item_model',
                        },
                    }
                ]
            }
        }
    )
    app_client.fake_os.count = AsyncMock(return_value={'count': 0})

    r = app_client.get('/curation/viz/projection')
    assert r.status_code == 200
    body = r.json()
    assert body['projection_version'] == 'umap_viz_v1'
    assert body['stale'] is False
    assert body['points'] == [
        {
            'crop_id': 'crop-a',
            'x': 0.5,
            'y': -0.5,
            'cluster_id': 42,
            'class_name': 'sedan',
            'class_source': 'item_model',
        }
    ]


def test_get_projection_max_points_caps_the_search_size(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    app_client.fake_os.get = AsyncMock(
        return_value={'_source': {'projection_version': 'umap_viz_v1', 'fitted_at': 't0'}}
    )
    app_client.fake_os.search = AsyncMock(return_value={'hits': {'hits': []}})
    app_client.fake_os.count = AsyncMock(return_value={'count': 0})

    r = app_client.get('/curation/viz/projection', params={'max_points': 25})
    assert r.status_code == 200
    search_kwargs = app_client.fake_os.search.call_args.kwargs
    assert search_kwargs['body']['size'] == 25


def test_get_projection_max_points_over_ceiling_422(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    r = app_client.get('/curation/viz/projection', params={'max_points': 10_000_000})
    assert r.status_code == 422


# =============================================================================
# Hard requirement (plan Part A.2): GET must never be able to trigger a fit.
# =============================================================================


def test_get_projection_never_imports_or_calls_fit(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monkeypatch the only fit-capable function in embedding_viz.py to
    explode, then drive the full router GET path end-to-end. If the GET
    handler were somehow reachable into a fit, this test fails loudly
    instead of silently passing."""
    from src.services.curation import embedding_viz

    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')

    def _boom(*args, **kwargs):
        raise AssertionError('GET /curation/viz/projection must never call fit_projection')

    monkeypatch.setattr(embedding_viz, 'fit_projection', _boom)
    monkeypatch.setattr(embedding_viz, '_build_reducer', _boom)
    monkeypatch.setattr(embedding_viz, 'run_projection_job', _boom)
    monkeypatch.setattr(embedding_viz, 'start_job', _boom)

    app_client.fake_os.get = AsyncMock(
        return_value={'_source': {'projection_version': 'umap_viz_v1', 'fitted_at': 't0'}}
    )
    app_client.fake_os.search = AsyncMock(return_value={'hits': {'hits': []}})
    app_client.fake_os.count = AsyncMock(return_value={'count': 0})

    r = app_client.get('/curation/viz/projection')
    assert r.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
