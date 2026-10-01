"""Route mounting (projects_plan.md §3.2, owner decision "no backwards
compatibility", review delta 1): every curation route is scoped under
``/curation/projects/{project}`` or is one of the global routes; there is
no unscoped alias, so an unscoped curation path is a 404, and the global
routes answer with nothing bound."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient


API = '/curation'
SCOPED = f'{API}/projects/{{project}}'

# Unscoped curation paths that are global by nature (served in OpenAPI).
# The P3 lifecycle mutations (create/patch/archive/unarchive/
# clone_settings/delete) act *on* a project, not *within* one, so they
# stay on global_router like the P1 reads. ``/stats`` is conceptually
# scoped (delta 9c: it reads the bound project's own counts) but is
# registered directly on global_router with its own `bind_path_project`
# dependency rather than through the `{SCOPED}` double-mount, since
# lifecycle routes never go through the scoped/alias split.
GLOBAL_ROUTES = frozenset(
    {
        ('GET', f'{API}/projects'),
        ('POST', f'{API}/projects'),
        ('GET', f'{API}/projects/{{project}}'),
        ('PATCH', f'{API}/projects/{{project}}'),
        ('DELETE', f'{API}/projects/{{project}}'),
        ('POST', f'{API}/projects/{{project}}/archive'),
        ('POST', f'{API}/projects/{{project}}/unarchive'),
        ('POST', f'{API}/projects/{{project}}/clone_settings'),
        ('GET', f'{API}/projects/{{project}}/stats'),
        ('POST', f'{API}/projects/combine/preview'),
        ('POST', f'{API}/projects/combine'),
        ('GET', f'{API}/projects/combine/{{job_id}}'),
        ('POST', f'{API}/projects/combine/{{job_id}}/cancel'),
        ('POST', f'{API}/projects/combine/{{job_id}}/resume'),
        ('GET', f'{API}/health'),
        ('GET', f'{API}/events'),
        # W9: the VLM endpoint registry and local model catalog are
        # deployment-wide; only the per-project activation is scoped.
        ('GET', f'{API}/vlm/catalog'),
        ('GET', f'{API}/vlm/local'),
        ('POST', f'{API}/vlm/local/select'),
        ('DELETE', f'{API}/vlm/local/select'),
        ('GET', f'{API}/vlm/endpoints'),
        ('GET', f'{API}/vlm/endpoints/schema'),
        ('POST', f'{API}/vlm/endpoints/validate'),
        ('GET', f'{API}/vlm/endpoints/{{name}}'),
        ('GET', f'{API}/vlm/endpoints/{{name}}/revisions'),
        ('GET', f'{API}/vlm/endpoints/{{name}}/revisions/{{revision}}'),
        ('POST', f'{API}/vlm/endpoints'),
        ('POST', f'{API}/vlm/endpoints/{{name}}/clone'),
        ('PUT', f'{API}/vlm/endpoints/{{name}}'),
        ('DELETE', f'{API}/vlm/endpoints/{{name}}'),
        ('POST', f'{API}/vlm/endpoints/{{name}}/probe'),
    }
)


def _curation_routes() -> list[APIRoute]:
    from src.main import app

    return [r for r in app.routes if isinstance(r, APIRoute) and r.path.startswith(f'{API}')]


def test_every_curation_route_is_scoped_or_global() -> None:
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
        unexpected.append(f'{sorted(route.methods)} {route.path}')
    assert unexpected == []


def test_no_route_is_hidden_from_openapi() -> None:
    hidden = [
        f'{sorted(r.methods)} {r.path}' for r in _curation_routes() if not r.include_in_schema
    ]
    assert hidden == []


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
    for slug, error in (('combine', 'combine_not_found'), ('no-such-project', 'project_not_found')):
        response = client.get(f'{API}/projects/{slug}/health')
        assert response.status_code == 404
        # `combine` is reserved: `/projects/combine/<x>` is the combine job
        # lookup (an unknown id), never a scoped route of a project.
        assert response.json()['detail']['error'] == error


def test_unscoped_curation_path_is_404(client: TestClient) -> None:
    """No alias: the old unscoped form of a scoped route does not exist."""
    for path in ('/crops', '/classes', '/methods', '/train/runs', '/settings'):
        assert client.get(f'{API}{path}').status_code == 404, path


def test_global_routes_answer_with_nothing_bound(client: TestClient) -> None:
    """Requests start unbound (tests/conftest.py runs each in a fresh
    context, as uvicorn does); the global routes must not need a binding."""
    from src.config.project_context import is_project_bound

    assert client.get(f'{API}/projects').status_code == 200
    assert client.get(f'{API}/health').status_code == 200
    from src.main import app

    events = next(r for r in app.routes if isinstance(r, APIRoute) and r.path == f'{API}/events')
    assert 'project' not in {p.name for p in events.dependant.path_params}
    assert is_project_bound()  # the test's own binding is untouched


def test_request_does_not_inherit_the_test_binding() -> None:
    """A route that forgets to bind must fail in tests the way it fails
    under uvicorn: the test's autouse binding never reaches the app."""
    from fastapi import FastAPI

    from src.config.project_context import is_project_bound

    app = FastAPI()

    @app.get('/probe')
    async def _probe_async() -> dict[str, bool]:
        return {'bound': is_project_bound()}

    @app.get('/probe_sync')
    def _probe_sync() -> dict[str, bool]:
        return {'bound': is_project_bound()}

    assert is_project_bound()
    client = TestClient(app)
    assert client.get('/probe').json() == {'bound': False}
    assert client.get('/probe_sync').json() == {'bound': False}
    with TestClient(app) as managed:
        assert managed.get('/probe').json() == {'bound': False}


def test_a_failed_project_does_not_bind(client: TestClient) -> None:
    """A ``failed`` project's index set may be half-created: 409, never a
    (writable) binding."""
    import dataclasses

    from src.config.curation import base_curation_config, base_curation_config as _base
    from src.config.projects import new_project_record, resources_for_new
    from src.services.projects import registry as registry_mod

    failed = dataclasses.replace(
        new_project_record('default', _base()),
        slug='broken',
        status='failed',
        resources=resources_for_new('broken', base_curation_config()),
    )
    registry_mod.get_project_registry()._by_slug['broken'] = failed
    response = client.get(f'{API}/projects/broken/classes')
    assert response.status_code == 409
    assert response.json()['detail']['error'] == 'project_failed'


def test_shell_and_cli_callers_use_scoped_curation_paths() -> None:
    """Re-review R4: the installer, deploy CLI, model-setup lib and the
    Makefile call only global or project-scoped curation paths."""
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    files = [
        *sorted((repo / 'scripts' / 'lib').glob('*.sh')),
        *sorted((repo / 'scripts').glob('*.sh')),
        repo / 'openprocessor',
        repo / 'setup-openprocessor.sh',
        repo / 'Makefile',
    ]
    allowed = re.compile(
        r'/curation/(projects(/|\b)|health\b|events\b|vlm/(endpoints/[^/]+/probe|local)\b)'
    )
    offenders = [
        f'{path.relative_to(repo)}:{lineno}: {line.strip()}'
        for path in files
        if path.is_file()
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1)
        for match in re.finditer(r'/curation/[A-Za-z_]', line)
        if not allowed.match(line[match.start() :])
        and 'scripts/curation/' not in line[max(0, match.start() - 8) : match.end() + 1]
    ]
    assert offenders == []
