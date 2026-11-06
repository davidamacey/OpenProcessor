"""The job list and read routes: a project sees only its jobs; pages are stable (3A-9)."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import uuid
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from fastapi import Depends, FastAPI
from fixtures.job_rows import insert_row, make_project, sql

from src.config.curation_accessors import base_curation_config
from src.config.project_context import set_bound_project
from src.config.projects import ProjectRecord, resources_for_new
from src.core.db_session import install_database_error_handlers
from src.routers.curation._project_deps import install_project_exception_handlers


pytestmark = pytest.mark.db

CONTRACT = Path(__file__).resolve().parents[2] / 'contracts' / 'openapi' / 'curation.json'


def record_for(project: uuid.UUID, slug: str) -> ProjectRecord:
    now = dt.datetime.now(dt.UTC).isoformat()
    record = ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )
    return dataclasses.replace(record, id=project)


@pytest.fixture
def app(pg, pg_env):
    """The real job routes; the project binding is the one thing replaced: it binds by slug."""
    from src.routers import curation as curation_package
    from src.routers.curation._project_deps import bind_path_project
    from src.routers.jobs import router as jobs_router

    records: dict[str, ProjectRecord] = {}
    application = FastAPI()

    async def bind(project: str) -> None:
        set_bound_project(records[project])

    application.dependency_overrides[bind_path_project] = bind
    application.include_router(
        curation_package.router,
        prefix='/curation/projects/{project}',
        dependencies=[Depends(bind_path_project)],
    )
    application.include_router(jobs_router)
    install_database_error_handlers(application)
    install_project_exception_handlers(application)
    application.state.records = records
    return application


@pytest_asyncio.fixture
async def client(app):
    from src.db.engine import dispose_engines

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as http:
        yield http
    await dispose_engines()


def project(app, pg, slug: str) -> uuid.UUID:
    pid = make_project(pg, slug)
    app.state.records[slug] = record_for(pid, slug)
    return pid


def url(slug: str, tail: str = '') -> str:
    return f'/curation/projects/{slug}/jobs{tail}'


@pytest.mark.asyncio
async def test_a_project_sees_only_its_own_jobs(app, pg, client) -> None:
    mine, other = project(app, pg, 'aa-mine'), project(app, pg, 'aa-other')
    mine_job = insert_row(pg, mine, 'running', kind='reprocess')
    insert_row(pg, other, 'running', kind='reprocess')
    insert_row(pg, None, 'queued', kind='project_delete')

    reply = await client.get(url('aa-mine'))
    assert reply.status_code == 200
    body = reply.json()
    assert [j['id'] for j in body['items']] == [str(mine_job)]
    job = body['items'][0]
    assert (job['kind'], job['project'], job['status']) == ('reprocess', 'aa-mine', 'running')
    assert job['params'] is None  # the list leaves parameters out


@pytest.mark.asyncio
async def test_one_job_has_its_params_and_another_projects_id_is_404(app, pg, client) -> None:
    mine, other = project(app, pg, 'ab-mine'), project(app, pg, 'ab-other')
    job = insert_row(pg, mine, 'completed', params=json.dumps({'a': 1}))
    foreign = insert_row(pg, other, 'completed')

    one = await client.get(url('ab-mine', f'/{job}'))
    assert one.status_code == 200
    assert one.json()['params'] == {'a': 1}
    for missing in (foreign, uuid.uuid4()):
        reply = await client.get(url('ab-mine', f'/{missing}'))
        assert reply.status_code == 404
        assert reply.json()['detail']['error'] == 'job_not_found'


@pytest.mark.asyncio
async def test_filters_by_kind_and_status(app, pg, client) -> None:
    pid = project(app, pg, 'ac-proj')
    wanted = insert_row(pg, pid, 'running', kind='probe')
    insert_row(pg, pid, 'running', kind='selection')
    insert_row(pg, pid, 'completed', kind='probe')
    reply = await client.get(url('ac-proj'), params={'kind': 'probe', 'status': 'running'})
    assert [j['id'] for j in reply.json()['items']] == [str(wanted)]
    both = await client.get(url('ac-proj'), params=[('kind', 'probe'), ('kind', 'selection')])
    assert len(both.json()['items']) == 3


@pytest.mark.asyncio
async def test_pagination_is_stable_while_new_jobs_arrive(app, pg, client) -> None:
    pid = project(app, pg, 'ad-proj')
    base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    original = [
        insert_row(pg, pid, 'completed', created_at=base + dt.timedelta(minutes=i))
        for i in range(7)
    ]
    first = (await client.get(url('ad-proj'), params={'limit': 3})).json()
    assert len(first['items']) == 3
    assert first['next_cursor']

    # new jobs land at the head between the two requests
    fresh = [insert_row(pg, pid, 'queued') for _ in range(2)]

    seen = [j['id'] for j in first['items']]
    cursor = first['next_cursor']
    while cursor:
        page = (await client.get(url('ad-proj'), params={'limit': 3, 'cursor': cursor})).json()
        seen += [j['id'] for j in page['items']]
        cursor = page['next_cursor']
    assert seen == [str(j) for j in reversed(original)]  # newest first, none twice, none missed
    assert not set(map(str, fresh)) & set(seen)
    head = (await client.get(url('ad-proj'), params={'limit': 100})).json()['items']
    assert {j['id'] for j in head[:2]} == set(map(str, fresh))


@pytest.mark.asyncio
async def test_an_unknown_cursor_is_422(app, pg, client) -> None:
    project(app, pg, 'ae-proj')
    reply = await client.get(url('ae-proj'), params={'cursor': 'not-a-cursor'})
    assert reply.status_code == 422
    assert reply.json()['detail']['error'] == 'invalid_cursor'


@pytest.mark.asyncio
async def test_the_deployment_list_holds_only_jobs_without_a_project(app, pg, client) -> None:
    pid = project(app, pg, 'af-proj')
    insert_row(pg, pid, 'running')
    sql(pg, 'DELETE FROM jobs WHERE project_id IS NULL')
    wide = insert_row(pg, None, 'queued', kind='directory_ingest')
    reply = await client.get('/jobs')
    assert reply.status_code == 200
    items = reply.json()['items']
    assert [j['id'] for j in items] == [str(wide)]
    assert items[0]['project'] is None


def test_the_routes_are_in_the_committed_contract() -> None:
    paths = json.loads(CONTRACT.read_text(encoding='utf-8'))['paths']
    assert 'get' in paths['/curation/projects/{project}/jobs']
    assert 'get' in paths['/curation/projects/{project}/jobs/{job_id}']
    assert 'post' not in paths['/curation/projects/{project}/jobs']
