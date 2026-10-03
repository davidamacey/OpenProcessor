"""Every item list, review, stats and cluster route takes the one shared item
filter and builds its query from it."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation import _common


SHARED_PARAMS = {
    'class_name',
    'exclude_class_name',
    'conf_min',
    'conf_max',
    'min_area',
    'max_area',
    'max_rank',
    'origin',
    'embedding_state',
    'review_status',
}

# Routes that filter items. The value is why a route is absent from the sweep
# when it has no item filter (kept here so a new list route has to choose).
FILTERED_ROUTES = [
    '/crops',
    '/review/{tab}',
    '/review/{tab}/locate',
    '/search/text',
    '/stats/classes',
    '/stats/dataset',
    '/clusters',
    '/regions',
]


class _RecordingOS:
    def __init__(self) -> None:
        self.bodies: list[str] = []

    async def search(self, *, index: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:  # noqa: ARG002
        self.bodies.append(json.dumps(body))
        return {'hits': {'total': {'value': 0}, 'hits': []}, 'aggregations': {}}

    async def count(self, *, index: str, body: dict[str, Any] | None = None) -> dict[str, Any]:  # noqa: ARG002
        self.bodies.append(json.dumps(body or {}))
        return {'count': 0}

    async def msearch(self, *, body: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        self.bodies.append(json.dumps(body))
        return {'responses': []}

    async def get(self, **_: Any) -> dict[str, Any]:
        return {'found': False}


@pytest.fixture
def fake_os() -> _RecordingOS:
    return _RecordingOS()


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, fake_os: _RecordingOS) -> FastAPI:
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', {'default'})
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return app


@pytest.mark.parametrize('path', FILTERED_ROUTES)
def test_openapi_declares_the_shared_filter_on_the_route(app: FastAPI, path: str) -> None:
    spec = app.openapi()
    operation = spec['paths'][f'{_common.config.api_prefix}/projects/{{project}}{path}']['get']
    declared = {p['name'] for p in operation['parameters']}
    assert declared >= SHARED_PARAMS, sorted(SHARED_PARAMS - declared)


PARAMS = {
    'class_name': 'hot dog',
    'min_area': '0.25',
    'embedding_state': 'failed',
    'origin': 'import',
    'review_status': 'validated',
}


@pytest.mark.parametrize(
    'path',
    ['/crops', '/review/all', '/stats/classes', '/stats/dataset', '/clusters', '/regions'],
)
def test_the_filter_reaches_the_query(
    app: FastAPI, fake_os: _RecordingOS, request: pytest.FixtureRequest, path: str
) -> None:
    if path == '/regions':
        request.getfixturevalue('reference_region_profile')
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get(f'{_common.config.api_prefix}/projects/default{path}', params=PARAMS)
    assert r.status_code < 500, r.text
    sent = ' '.join(fake_os.bodies)
    for needle in ('hot dog', 'crop_area_norm', 'embedding_state', 'import_ids', 'class_validated'):
        assert needle in sent, (path, needle)


@pytest.mark.parametrize('path', ['/crops', '/stats/classes', '/clusters'])
def test_a_malformed_band_is_a_400(app: FastAPI, path: str) -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get(
            f'{_common.config.api_prefix}/projects/default{path}',
            params={'min_area': '0.9', 'max_area': '0.1'},
        )
    assert r.status_code == 400, r.text


def test_the_review_catalog_advertises_the_shared_filters(app: FastAPI) -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        tabs = client.get(f'{_common.config.api_prefix}/projects/default/review/tabs').json()[
            'tabs'
        ]
    for tab in tabs:
        assert SHARED_PARAMS - {'max_rank'} <= set(tab['filters']), tab['id']
