"""Route mounting (projects_plan.md §3.2, review delta 1): every curation
route is scoped under ``/curation/projects/{project}`` or global; the
unscoped ``default`` alias is hidden from OpenAPI; the global ``/health``
and ``/events`` win over the alias."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient


API = '/curation'
SCOPED = f'{API}/projects/{{project}}'

# Unscoped curation paths that are global by nature (served in OpenAPI).
GLOBAL_ROUTES = frozenset(
    {
        ('GET', f'{API}/projects'),
        ('GET', f'{API}/projects/{{project}}'),
        ('GET', f'{API}/health'),
        ('GET', f'{API}/events'),
    }
)


def _curation_routes() -> list[APIRoute]:
    from src.main import app

    return [r for r in app.routes if isinstance(r, APIRoute) and r.path.startswith(f'{API}')]


def test_every_curation_route_is_scoped_global_or_hidden_alias() -> None:
    from src.main import app  # noqa: F401 - assembling the app registers the global routes
    from src.routers.curation.projects import global_router

    global_endpoints = {r.endpoint for r in global_router.routes}
    unexpected = []
    for route in _curation_routes():
        if route.path.startswith(f'{SCOPED}/'):
            continue
        pairs = {(m, route.path) for m in route.methods}
        if route.endpoint in global_endpoints and pairs <= GLOBAL_ROUTES:
            continue
        if not route.include_in_schema:
            continue  # the unscoped `default` alias
        unexpected.append(f'{sorted(route.methods)} {route.path}')
    assert unexpected == []


def test_alias_mirrors_every_scoped_route_hidden_from_openapi() -> None:
    routes = _curation_routes()
    scoped = {
        (m, r.path[len(SCOPED) :])
        for r in routes
        if r.path.startswith(f'{SCOPED}/')
        for m in r.methods
    }
    alias = {
        (m, r.path[len(API) :])
        for r in routes
        if not r.include_in_schema and not r.path.startswith(f'{API}/projects')
        for m in r.methods
    }
    assert scoped == alias


def test_openapi_documents_only_scoped_or_global_paths() -> None:
    from src.main import app

    paths = app.openapi()['paths']
    for path, ops in paths.items():
        if not path.startswith(f'{API}/'):
            continue
        if path.startswith(f'{SCOPED}/'):
            for method, op in ops.items():
                params = {p['name']: p for p in op.get('parameters', []) if p['in'] == 'path'}
                assert 'project' in params, f'{method.upper()} {path} lacks the project param'
                assert params['project']['required'] is True
            continue
        documented = {(m.upper(), path) for m in ops}
        assert documented <= GLOBAL_ROUTES, f'unscoped path in OpenAPI: {path}'


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    from src.core.dependencies import get_async_triton
    from src.main import app
    from src.routers.curation import global_status, health

    async def _down(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {'reachable': False}

    monkeypatch.setattr(health, 'triton_status', _down)
    monkeypatch.setattr(global_status, 'triton_status', _down)
    monkeypatch.setattr(health, 'vlm_status', _down)
    monkeypatch.setattr(global_status, 'vlm_status', _down)

    async def _no_opensearch() -> dict[str, Any]:
        return {'reachable': False, 'detail': 'disabled in test'}

    monkeypatch.setattr(global_status, '_opensearch_reachable', _no_opensearch)

    class _NoOpenSearch:
        class indices:  # noqa: N801 - mirrors the client attribute
            @staticmethod
            async def exists(**_k: Any) -> bool:
                raise ConnectionError('disabled in test')

    from src.routers.curation._common import _raw_opensearch_dep
    from src.services.projects import registry as registry_mod

    async def _fresh(self: Any) -> None:
        return None

    monkeypatch.setattr(registry_mod.ProjectRegistry, 'ensure_fresh', _fresh)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: _NoOpenSearch()
    app.dependency_overrides[get_async_triton] = lambda: object()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(_raw_opensearch_dep, None)
        app.dependency_overrides.pop(get_async_triton, None)


def test_global_health_is_deployment_only(client: TestClient) -> None:
    body = client.get(f'{API}/health').json()
    assert set(body) == {
        'status',
        'triton',
        'opensearch',
        'vlm',
        'mlflow_public_url',
        'version',
        'api_version',
    }
    assert 'project' not in body


def test_scoped_health_carries_the_bound_project(client: TestClient) -> None:
    body = client.get(f'{API}/projects/default/health').json()
    assert body['project'] == 'default'
    assert {'region_profile', 'registry', 'opensearch'} <= set(body)


def test_unknown_or_reserved_slug_is_404_not_a_scoped_route(client: TestClient) -> None:
    for slug in ('combine', 'no-such-project'):
        response = client.get(f'{API}/projects/{slug}/health')
        assert response.status_code == 404
        assert response.json()['detail']['error'] == 'project_not_found'
