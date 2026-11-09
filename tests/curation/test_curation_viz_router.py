"""Tests for the UMAP visualization-only projection overlay: ``POST
/curation/viz/projection/rebuild`` + ``status``/``cancel`` lifecycle, and
``GET /curation/viz/projection``.

Mount the real curation router with OpenSearch stubbed and the job seam (``embedding_viz.start``,
``latest_job``, ``cancel_latest``) replaced; the queue, busy and run paths are covered by
``tests/curation/jobs_a``.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from config_store.memory_settings import current
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.core.db_session import get_db
    from src.routers.curation._common import _raw_opensearch_dep, router as curation_router

    async def space(_target):
        return SimpleNamespace(vector_field='pe_embedding')

    monkeypatch.setattr('src.services.curation.embedding_viz_read.embedding_space', space)
    monkeypatch.setattr('src.routers.curation.viz.current_project_id', lambda: 'project-id')

    fake_os = AsyncMock()
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os

    async def no_database():
        yield None

    app.dependency_overrides[get_db] = no_database
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


# =============================================================================
# Feature-flag gating
# =============================================================================


def test_rebuild_disabled_400(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_VIZ_PROJECTION_ENABLED', raising=False)
    r = app_client.post('/curation/projects/default/viz/projection/rebuild')
    assert r.status_code == 400
    assert 'disabled' in r.json()['detail']


def test_get_projection_disabled_400(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_VIZ_PROJECTION_ENABLED', raising=False)
    r = app_client.get('/curation/projects/default/viz/projection')
    assert r.status_code == 400
    assert 'disabled' in r.json()['detail']


def test_status_and_cancel_are_not_gated_by_the_flag(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same convention as scores_status/select_status: read-only lifecycle
    endpoints work regardless of the feature flag, so an operator can
    always see current state."""
    monkeypatch.delenv('OP_VIZ_PROJECTION_ENABLED', raising=False)
    _job_seam(monkeypatch, None)
    r_status = app_client.get('/curation/projects/default/viz/projection/status')
    assert r_status.status_code == 200
    assert r_status.json()['status'] == 'idle'

    r_cancel = app_client.post('/curation/projects/default/viz/projection/cancel')
    assert r_cancel.status_code == 200
    assert r_cancel.json()['cancelled'] is False


# =============================================================================
# POST /curation/viz/projection/rebuild — validation + job lifecycle
# =============================================================================


def test_rebuild_cluster_scope_without_cluster_uid_400(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    r = app_client.post(
        '/curation/projects/default/viz/projection/rebuild', params={'scope': 'cluster'}
    )
    assert r.status_code == 400
    assert 'cluster_uid' in r.json()['detail']


def test_rebuild_unknown_scope_422(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    r = app_client.post(
        '/curation/projects/default/viz/projection/rebuild', params={'scope': 'bogus'}
    )
    assert r.status_code == 422  # FastAPI Query(pattern=...) rejects before the handler runs


def _queued_view(scope: str = 'residual', cluster_uid: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id='j1',
        status='queued',
        started_at=None,
        finished_at=None,
        error_message=None,
        params={'scope': scope, 'cluster_uid': cluster_uid},
        progress={},
        result=None,
    )


def _job_seam(monkeypatch: pytest.MonkeyPatch, latest) -> AsyncMock:
    start = AsyncMock()
    monkeypatch.setattr('src.services.curation.embedding_viz.start', start)
    monkeypatch.setattr(
        'src.services.curation.latest_job.latest_job', AsyncMock(return_value=latest)
    )
    return start


def test_rebuild_queues_the_job_and_answers_202_with_its_state(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    start = _job_seam(monkeypatch, _queued_view())
    r = app_client.post(
        '/curation/projects/default/viz/projection/rebuild', params={'scope': 'residual'}
    )
    assert r.status_code == 202
    assert (r.json()['status'], r.json()['scope']) == ('queued', 'residual')
    assert start.await_args.kwargs['scope'] == 'residual'


def test_rebuild_cluster_scope_passes_cluster_uid_through(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    start = _job_seam(monkeypatch, _queued_view('cluster', '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11'))
    r = app_client.post(
        '/curation/projects/default/viz/projection/rebuild',
        params={'scope': 'cluster', 'cluster_uid': '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11'},
    )
    assert r.status_code == 202
    assert (r.json()['scope'], r.json()['cluster_uid']) == (
        'cluster',
        '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11',
    )
    assert start.await_args.kwargs['cluster_uid'] == uuid.UUID(
        '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11'
    )


def test_cancel_reports_whether_a_job_was_cancelled(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _job_seam(monkeypatch, _queued_view())
    cancel = AsyncMock(return_value=True)
    monkeypatch.setattr('src.services.curation.latest_job.cancel_latest', cancel)
    r = app_client.post('/curation/projects/default/viz/projection/cancel')
    assert r.status_code == 200
    assert r.json()['cancelled'] is True


# =============================================================================
# GET /curation/viz/projection — cached-coordinates-only serving
# =============================================================================


def test_get_projection_not_built(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    current().projection = None

    r = app_client.get('/curation/projects/default/viz/projection')
    assert r.status_code == 404
    assert r.json()['error'] == 'projection_not_built'
    app_client.fake_os.search.assert_not_called()


def test_get_projection_read_failure_is_503_not_not_built(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    from src.services.curation.embedding_viz_read import ProjectionUnavailableError

    async def _down() -> None:
        raise ProjectionUnavailableError('database is down')

    monkeypatch.setattr('src.services.curation.embedding_viz_read._load_run_metadata', _down)

    r = app_client.get('/curation/projects/default/viz/projection')
    assert r.status_code == 503
    assert r.json()['error'] == 'projection_unavailable'


def test_get_projection_serves_cached_points(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    current().projection = {
        'projection_version': 'umap_viz_v1',
        'fitted_at': '2026-09-11T00:00:00+00:00',
    }
    app_client.fake_os.search = AsyncMock(
        return_value={
            'hits': {
                'hits': [
                    {
                        '_id': 'crop-a',
                        '_source': {
                            'viz_x': 0.5,
                            'viz_y': -0.5,
                            'cluster_uid': '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11',
                            'class_name': 'sedan',
                            'class_source': 'item_model',
                        },
                    }
                ]
            }
        }
    )
    app_client.fake_os.count = AsyncMock(return_value={'count': 0})

    r = app_client.get('/curation/projects/default/viz/projection')
    assert r.status_code == 200
    body = r.json()
    assert body['projection_version'] == 'umap_viz_v1'
    assert body['stale'] is False
    assert body['points'] == [
        {
            'crop_id': 'crop-a',
            'x': 0.5,
            'y': -0.5,
            'cluster_uid': '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11',
            'class_name': 'sedan',
            'class_source': 'item_model',
        }
    ]


def test_get_projection_max_points_caps_the_search_size(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    current().projection = {'projection_version': 'umap_viz_v1', 'fitted_at': 't0'}
    app_client.fake_os.search = AsyncMock(return_value={'hits': {'hits': []}})
    app_client.fake_os.count = AsyncMock(return_value={'count': 0})

    r = app_client.get('/curation/projects/default/viz/projection', params={'max_points': 25})
    assert r.status_code == 200
    search_kwargs = app_client.fake_os.search.call_args.kwargs
    assert search_kwargs['body']['size'] == 25


def test_get_projection_max_points_over_ceiling_422(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_PROJECTION_ENABLED', '1')
    r = app_client.get(
        '/curation/projects/default/viz/projection', params={'max_points': 10_000_000}
    )
    assert r.status_code == 422


# =============================================================================
# Hard requirement: GET must never be able to trigger a fit.
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
    monkeypatch.setattr(embedding_viz, 'start', _boom)

    current().projection = {'projection_version': 'umap_viz_v1', 'fitted_at': 't0'}
    app_client.fake_os.search = AsyncMock(return_value={'hits': {'hits': []}})
    app_client.fake_os.count = AsyncMock(return_value={'count': 0})

    r = app_client.get('/curation/projects/default/viz/projection')
    assert r.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
