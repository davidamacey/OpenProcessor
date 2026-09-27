"""The project lifecycle surface over HTTP: status codes, error bodies and
the typed OpenAPI shapes a client generates from (projects_plan.md, "P3
finish-pass inputs")."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation import projects as projects_router
from src.services.projects import lifecycle
from src.services.projects.registry import (
    ProjectRegistry,
    get_record_with_seq,
    set_project_registry,
    write_record,
)

from .conftest import FakeLifecycleOpenSearch, seed_default_project


if TYPE_CHECKING:
    from src.config.projects import ProjectStatus


class _CountingOpenSearch(FakeLifecycleOpenSearch):
    """``count`` honours a ``term`` query, so a validated count is real."""

    async def count(self, *, index: str, body: Any = None) -> dict[str, Any]:
        docs = self.indexes.get(index, [])
        term = ((body or {}).get('query') or {}).get('term')
        if not term:
            return {'count': len(docs)}
        ((field, value),) = term.items()
        return {'count': sum(1 for d in docs if d.get(field) == value)}


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import src.config.curation as curation_mod
    from src.services.projects import capacity as capacity_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)
    yield
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)


@pytest.fixture(autouse=True)
def _noop_ensure_indexes():
    with patch('src.routers.curation._common._ensure_indexes', new=AsyncMock()):
        yield


@pytest.fixture
def client(monkeypatch) -> _CountingOpenSearch:
    fake = _CountingOpenSearch()
    set_project_registry(ProjectRegistry(lambda: fake))

    async def _make() -> _CountingOpenSearch:
        return fake

    monkeypatch.setattr(projects_router, 'make_curation_opensearch', _make)
    return fake


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(projects_router.global_router, prefix='/curation')
    return app


@pytest.fixture
def http(client: _CountingOpenSearch) -> TestClient:
    return TestClient(_app())


def _create(slug: str, client: _CountingOpenSearch) -> Any:
    record, _ = asyncio.run(lifecycle.create_project(client, slug=slug, display_name=slug))
    return record


def _stored(client: _CountingOpenSearch, slug: str) -> Any:
    record = asyncio.run(get_record_with_seq(client, slug))[0]
    assert record is not None
    return record


def _set_status(client: _CountingOpenSearch, slug: str, status: ProjectStatus) -> None:
    async def _run() -> None:
        record, seq, term = await get_record_with_seq(client, slug)
        assert record is not None
        await write_record(
            client, replace(record, status=status), if_seq_no=seq, if_primary_term=term
        )
        await projects_router.get_project_registry().ensure_fresh()

    asyncio.run(_run())


def _revision(http: TestClient, slug: str) -> int:
    r = http.get(f'/curation/projects/{slug}')
    assert r.status_code == 200, r.text
    return r.json()['revision']


# --- 2. typed DELETE response + typed ProjectsResponse.capacity -------------


def _schema(app: FastAPI) -> dict[str, Any]:
    return app.openapi()


def test_delete_route_declares_typed_responses() -> None:
    spec = _schema(_app())
    responses = spec['paths']['/curation/projects/{project}']['delete']['responses']
    assert responses['200']['content']['application/json']['schema'] == {
        '$ref': '#/components/schemas/DeleteDryRunResponse'
    }
    assert responses['202']['content']['application/json']['schema'] == {
        '$ref': '#/components/schemas/ProjectLifecycleResponse'
    }


def test_projects_response_capacity_is_a_typed_model() -> None:
    schemas = _schema(_app())['components']['schemas']
    capacity = schemas['ProjectsResponse']['properties']['capacity']
    assert {'$ref': '#/components/schemas/ProjectCapacityWire'} in capacity['anyOf']
    fields = set(schemas['ProjectCapacityWire']['properties'])
    assert {'status', 'active_shards', 'per_project_shards', 'soft_limit', 'hard_limit'} <= fields


# --- 3. 409 shard_budget_exceeded carries capacity on the wire --------------


def test_shard_budget_exceeded_409_carries_capacity(
    http: TestClient, client: _CountingOpenSearch
) -> None:
    async def _blocked(method, url, params=None, **kwargs):
        if url == '/_cluster/health':
            return {'active_shards': 999, 'number_of_data_nodes': 1}
        if url == '/_cluster/settings':
            return {'persistent': {'cluster.max_shards_per_node': 1000}, 'transient': {}}
        if url == '/_nodes/stats/jvm':
            return {'nodes': {'n1': {'jvm': {'mem': {'heap_max_in_bytes': 1024**3}}}}}
        raise NotImplementedError(url)

    client.transport.perform_request = _blocked  # type: ignore[method-assign]
    r = http.post('/curation/projects', json={'slug': 'overbudget', 'display_name': 'x'})
    assert r.status_code == 409, r.text
    detail = r.json()['detail']
    assert detail['error'] == 'shard_budget_exceeded'
    assert detail['capacity']['status'] == 'blocked'
    assert detail['capacity']['active_shards'] == 999
    assert detail['capacity']['hard_limit'] == 1000


def test_error_detail_documents_a_typed_capacity() -> None:
    schemas = _schema(_app())['components']['schemas']
    capacity = schemas['ConfigErrorDetail']['properties']['capacity']
    assert {'$ref': '#/components/schemas/ProjectCapacityWire'} in capacity['anyOf']


# --- 4. archivable / unarchivable + server-side transitions -----------------


def test_summary_serves_archivable_and_unarchivable(
    http: TestClient, client: _CountingOpenSearch
) -> None:
    _create('alpha', client)
    _create('beta', client)
    active = http.get('/curation/projects/alpha').json()
    assert active['archivable'] is True
    assert active['unarchivable'] is False
    r = http.post('/curation/projects/alpha/archive', json={'expected_revision': 1})
    assert r.status_code == 200, r.text
    archived = r.json()['project']
    assert archived['archivable'] is False
    assert archived['unarchivable'] is True


@pytest.mark.parametrize('status', ['building', 'failed', 'deleting', 'archived'])
def test_archive_refuses_a_non_active_project(
    status: ProjectStatus, http: TestClient, client: _CountingOpenSearch
) -> None:
    _create('alpha', client)
    _create('beta', client)
    _set_status(client, 'alpha', status)
    rev = _stored(client, 'alpha').revision
    r = http.post('/curation/projects/alpha/archive', json={'expected_revision': rev})
    assert r.status_code == 409, r.text
    assert r.json()['detail']['error'] == 'invalid_transition'
    assert _stored(client, 'alpha').status == status


@pytest.mark.parametrize('status', ['active', 'building', 'failed', 'deleting'])
def test_unarchive_refuses_a_non_archived_project(
    status: ProjectStatus, http: TestClient, client: _CountingOpenSearch
) -> None:
    _create('alpha', client)
    _set_status(client, 'alpha', status)
    rev = _stored(client, 'alpha').revision
    r = http.post('/curation/projects/alpha/unarchive', json={'expected_revision': rev})
    assert r.status_code == 409, r.text
    assert r.json()['detail']['error'] == 'invalid_transition'
    assert _stored(client, 'alpha').status == status


def test_building_and_failed_are_neither_archivable_nor_unarchivable(
    http: TestClient, client: _CountingOpenSearch
) -> None:
    _create('alpha', client)
    statuses: tuple[ProjectStatus, ...] = ('building', 'failed')
    for status in statuses:
        _set_status(client, 'alpha', status)
        body = http.get('/curation/projects?include_archived=true').json()
        (alpha,) = [p for p in body['projects'] if p['slug'] == 'alpha']
        assert alpha['archivable'] is False
        assert alpha['unarchivable'] is False


# --- 5. a refused clone_settings never changes the revision ----------------


@pytest.mark.parametrize(
    ('body', 'status', 'code'),
    [
        ({'from': 'nope', 'axes': ['settings_defaults']}, 404, 'project_not_found'),
        ({'from': 'alpha', 'axes': ['bogus']}, 422, 'combine_invalid'),
        ({'from': 'alpha', 'axes': ['classes']}, 409, 'target_not_empty'),
    ],
)
def test_refused_clone_settings_keeps_the_revision(
    body: dict[str, Any], status: int, code: str, http: TestClient, client: _CountingOpenSearch
) -> None:
    _create('alpha', client)
    beta = _create('beta', client)
    from src.config.curation import IndexRole

    client.indexes[beta.resources.indexes[IndexRole.ITEMS]] = [{'crop_id': 'b-1'}]
    before = _revision(http, 'beta')
    r = http.post(
        '/curation/projects/beta/clone_settings', json={**body, 'expected_revision': before}
    )
    assert r.status_code == status, r.text
    assert r.json()['detail']['error'] == code
    assert _revision(http, 'beta') == before


def test_accepted_clone_settings_bumps_the_revision_once(
    http: TestClient, client: _CountingOpenSearch
) -> None:
    _create('alpha', client)
    _create('beta', client)
    before = _revision(http, 'beta')
    r = http.post(
        '/curation/projects/beta/clone_settings',
        json={'from': 'alpha', 'axes': ['settings_defaults'], 'expected_revision': before},
    )
    assert r.status_code == 200, r.text
    assert r.json()['project']['revision'] == before + 1
    assert _revision(http, 'beta') == before + 1


def test_clone_settings_revision_conflict_changes_nothing(
    http: TestClient, client: _CountingOpenSearch
) -> None:
    _create('alpha', client)
    _create('beta', client)
    before = _revision(http, 'beta')
    r = http.post(
        '/curation/projects/beta/clone_settings',
        json={'from': 'alpha', 'expected_revision': before + 5},
    )
    assert r.status_code == 409, r.text
    assert r.json()['detail']['error'] == 'revision_conflict'
    assert _revision(http, 'beta') == before


# --- 6. counts.validated is computed, the same way on every route ----------


def test_counts_validated_is_the_same_number_on_every_route(
    http: TestClient, client: _CountingOpenSearch
) -> None:
    alpha = _create('alpha', client)
    _create('beta', client)
    from src.config.curation import IndexRole

    client.indexes[alpha.resources.indexes[IndexRole.ITEMS]] = [
        {'crop_id': 'a1', 'class_validated': True},
        {'crop_id': 'a2', 'class_validated': True},
        {'crop_id': 'a3', 'class_validated': False},
    ]
    listed = http.get('/curation/projects').json()
    (row,) = [p for p in listed['projects'] if p['slug'] == 'alpha']
    got = http.get('/curation/projects/alpha').json()
    patched = http.patch(
        '/curation/projects/alpha', json={'description': 'x', 'expected_revision': 1}
    ).json()
    stats = http.get('/curation/projects/alpha/stats').json()
    assert row['counts']['validated'] == 2
    assert got['counts']['validated'] == 2
    assert patched['project']['counts']['validated'] == 2
    assert stats['counts']['validated'] == 2


def test_counts_validated_is_null_on_every_route_when_uncountable(
    http: TestClient, client: _CountingOpenSearch, monkeypatch
) -> None:
    _create('alpha', client)
    _create('beta', client)

    async def _down(*_a: Any, **_kw: Any) -> Any:
        raise ConnectionError('opensearch down')

    monkeypatch.setattr(client, 'count', _down)
    listed = http.get('/curation/projects').json()
    (row,) = [p for p in listed['projects'] if p['slug'] == 'alpha']
    stats = http.get('/curation/projects/alpha/stats').json()
    assert row['counts']['validated'] is None
    assert http.get('/curation/projects/alpha').json()['counts']['validated'] is None
    assert stats['counts']['validated'] is None


# --- 7. default is archive-only ---------------------------------------------


def test_delete_default_dry_run_200_and_real_409(
    http: TestClient, client: _CountingOpenSearch
) -> None:
    asyncio.run(seed_default_project(client))
    _create('alpha', client)
    dry = http.delete('/curation/projects/default', params={'dry_run': 'true'})
    assert dry.status_code == 200, dry.text
    assert dry.json()['blocking'] == ['project_protected']

    for params in ({}, {'confirm': 'default'}, {'confirm': 'default', 'force': 'true'}):
        real = http.delete('/curation/projects/default', params=params)
        assert real.status_code == 409, (params, real.text)
        assert real.json()['detail']['error'] == 'project_protected'
    assert _stored(client, 'default').status == 'active'


# --- 8. the dry run's last_active_project matches enforcement ---------------


def test_last_active_project_excludes_archived_in_dry_run_and_enforcement(
    http: TestClient, client: _CountingOpenSearch
) -> None:
    _create('alpha', client)
    _create('beta', client)
    r = http.post('/curation/projects/alpha/archive', json={'expected_revision': 1})
    assert r.status_code == 200, r.text

    dry = http.delete('/curation/projects/beta', params={'dry_run': 'true'})
    assert dry.status_code == 200, dry.text
    assert 'last_active_project' in dry.json()['blocking']

    real = http.delete('/curation/projects/beta', params={'confirm': 'beta'})
    assert real.status_code == 409, real.text
    assert real.json()['detail']['error'] == 'last_active_project'

    archive = http.post('/curation/projects/beta/archive', json={'expected_revision': 1})
    assert archive.status_code == 409, archive.text
    assert archive.json()['detail']['error'] == 'last_active_project'


def test_dry_run_and_delete_agree_when_another_project_is_active(
    http: TestClient, client: _CountingOpenSearch
) -> None:
    _create('alpha', client)
    _create('beta', client)
    dry = http.delete('/curation/projects/beta', params={'dry_run': 'true'})
    assert dry.json()['blocking'] == []
    real = http.delete('/curation/projects/beta', params={'confirm': 'beta'})
    assert real.status_code == 202, real.text


# --- 9. malformed or unknown slug -> 404 project_not_found ------------------


@pytest.mark.parametrize('slug', ['Bad_Slug', 'x', 'UPPER', 'has.dot', 'a' * 80, 'nope'])
def test_get_malformed_or_unknown_slug_is_404(slug: str, http: TestClient) -> None:
    r = http.get(f'/curation/projects/{slug}')
    assert r.status_code == 404, r.text
    assert r.json()['detail']['error'] == 'project_not_found'


@pytest.mark.parametrize(
    ('method', 'suffix', 'body'),
    [
        ('patch', '', {'expected_revision': 1}),
        ('post', '/archive', {'expected_revision': 1}),
        ('post', '/unarchive', {'expected_revision': 1}),
        ('post', '/clone_settings', {'from': 'alpha', 'expected_revision': 1}),
        ('delete', '', None),
        ('get', '/stats', None),
    ],
)
def test_lifecycle_routes_malformed_slug_is_404(
    method: str, suffix: str, body: Any, http: TestClient
) -> None:
    kwargs: dict[str, Any] = {} if body is None else {'json': body}
    if method == 'delete':
        kwargs['params'] = {'dry_run': 'true'}
    r = http.request(method.upper(), f'/curation/projects/Bad_Slug{suffix}', **kwargs)
    assert r.status_code == 404, r.text
    assert r.json()['detail']['error'] == 'project_not_found'
