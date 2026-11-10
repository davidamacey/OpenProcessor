"""Routes typed in lane typed-responses-3 name their response model in the OpenAPI contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


CONTRACT = Path(__file__).resolve().parents[1] / 'contracts' / 'openapi' / 'curation.json'
PREFIX = '/curation/projects/{project}'

TYPED = [
    ('post', '/events/publish', '200', 'EventPublishAck'),
    ('get', '/events/stats', '200', 'EventStreamStats'),
    ('post', '/scores/compute', '202', 'ScoresState'),
    ('get', '/scores/status', '200', 'ScoresState'),
    ('post', '/scores/cancel', '200', 'ScoresCancelResponse'),
    ('get', '/scores/coverage', '200', 'ScoresCoverageResponse'),
    ('post', '/viz/projection/rebuild', '202', 'VizProjectionState'),
    ('get', '/viz/projection/status', '200', 'VizProjectionState'),
    ('post', '/viz/projection/cancel', '200', 'VizProjectionCancelResponse'),
    ('post', '/cluster/umap/rebuild', '200', 'UmapRebuildResponse'),
    ('get', '/regions/statuses', '200', 'RegionStatusCatalog'),
    ('get', '/regions/clusters', '200', 'RegionClusterList'),
    ('get', '/review/new_class_proposals/summary', '200', 'NewClassProposalSummary'),
    ('post', '/pipeline/auto_label', '200', 'AutoLabelRunSummary'),
]


@pytest.mark.parametrize(('method', 'path', 'status', 'model'), TYPED)
def test_response_schema_names_the_model(method: str, path: str, status: str, model: str) -> None:
    doc = json.loads(CONTRACT.read_text())
    operation = doc['paths'][PREFIX + path][method]
    schema = operation['responses'][status]['content']['application/json']['schema']
    assert schema['$ref'].rsplit('/', 1)[1] == model
    assert doc['components']['schemas'][model].get('properties')


@pytest.mark.parametrize('path', ['/events', '/pipeline/events'])
def test_event_streams_document_their_frames(path: str) -> None:
    doc = json.loads(CONTRACT.read_text())
    ok = doc['paths'][PREFIX + path]['get']['responses']['200']
    assert 'text/event-stream' in ok['content']
    assert 'server-sent' in ok['description'].lower()


def _client(router_module: Any, *dependency_overrides: tuple[Any, Any]) -> Any:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(router_module.router)
    for dep, override in dependency_overrides:
        app.dependency_overrides[dep] = override
    return TestClient(app)


def test_region_status_catalog_body_is_kept_whole() -> None:
    from src.config.region_state import region_status_catalog
    from src.routers.curation import regions

    body = _client(regions).get('/regions/statuses').json()
    assert body == json.loads(json.dumps(region_status_catalog()))


def test_umap_no_residuals_branch_keeps_exactly_its_keys(monkeypatch) -> None:
    from src.core.dependencies import get_curation_opensearch
    from src.routers.curation import umap

    async def fake(_client: Any) -> dict[str, Any]:
        return {'status': 'no_residuals', 'n_residuals': 0, 'refit': False}

    monkeypatch.setattr(umap, 'umap_rebuild', fake)
    client = _client(umap, (get_curation_opensearch, lambda: object()))
    assert client.post('/cluster/umap/rebuild').json() == {
        'status': 'no_residuals',
        'n_residuals': 0,
        'refit': False,
    }


def test_events_stats_keeps_every_key(monkeypatch) -> None:
    from src.routers.curation import events

    stats = {
        'subscribers': 2,
        'clients_dropped': 1,
        'listener_running': True,
        'pubsub_connected': False,
    }

    class Listener:
        def stats(self, _project: str | None) -> dict[str, Any]:
            return stats

    monkeypatch.setattr(events, 'get_listener', lambda: Listener())
    monkeypatch.setattr(
        events, 'get_curation_config', lambda: type('Cfg', (), {'project_slug': 'p'})()
    )
    assert _client(events).get('/events/stats').json() == stats


def test_scores_state_keeps_job_keys() -> None:
    from src.routers.curation._latest_job_models import ScoresCancelResponse
    from src.services.curation.item_scores import job as scores_job

    state = {**scores_job.state_dict(None), 'cancelled': False}
    assert ScoresCancelResponse(**state).model_dump() == state
