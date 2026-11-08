"""The embedding space routes against a real PostgreSQL: register, build, run status, isolation."""

from __future__ import annotations

import dataclasses
import datetime as dt
import uuid

import httpx
import pytest
import pytest_asyncio
from fastapi import Depends, FastAPI
from fixtures.job_rows import make_project, sql

from src.config.curation_accessors import base_curation_config
from src.config.project_context import set_bound_project
from src.config.projects import ProjectRecord, resources_for_new
from src.core.db_session import install_database_error_handlers
from src.db.repos import embedding_spaces as repo
from src.routers.curation._project_deps import install_project_exception_handlers
from tests.embedding_spaces._seed import shipped_engines


pytestmark = [
    pytest.mark.db,
    pytest.mark.asyncio,
    pytest.mark.usefixtures('bound_async_engine', 'api_async_engine'),
]

REGISTER = {'model_key': 'mobileclip2_s2', 'model_version': 1, 'target': 'item'}


class FakeCapacity:
    def __init__(self, docs: int = 1000, free: int = 10**12) -> None:
        self.docs, self.free = docs, free

    async def doc_count(self, _target) -> int:
        return self.docs

    async def free_bytes(self) -> int:
        return self.free


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
def capacity() -> FakeCapacity:
    return FakeCapacity()


@pytest.fixture
def app(pg, pg_env, catalog, capacity, producer, wakes, monkeypatch):
    from src.routers import curation as curation_package
    from src.routers.curation._project_deps import bind_path_project
    from src.routers.curation.embedding_spaces import _capacity

    monkeypatch.setattr(repo, 'ready_engine_contracts', lambda _session: shipped_engines())
    records: dict[str, ProjectRecord] = {}
    application = FastAPI()

    async def bind(project: str) -> None:
        set_bound_project(records[project])

    application.dependency_overrides[bind_path_project] = bind
    application.dependency_overrides[_capacity] = lambda: capacity
    application.include_router(
        curation_package.router,
        prefix='/curation/projects/{project}',
        dependencies=[Depends(bind_path_project)],
    )
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


def project(app, pg, name: str) -> str:
    slug = f'{name}{uuid.uuid4().hex[:6]}'
    app.state.records[slug] = record_for(make_project(pg, slug), slug)
    return slug


def jobs_of(pg, slug: str) -> int:
    return sql(
        pg,
        'SELECT count(*) FROM jobs j JOIN projects p ON p.id = j.project_id '
        "WHERE p.slug = %s AND j.kind = 'embedding_reindex'",
        slug,
    )[0][0]


def space_statuses(pg, slug: str) -> list[str]:
    rows = sql(
        pg,
        'SELECT s.status FROM embedding_spaces s JOIN projects p ON p.id = s.project_id '
        'WHERE p.slug = %s',
        slug,
    )
    return [r[0] for r in rows]


def url(slug: str, tail: str) -> str:
    return f'/curation/projects/{slug}/embedding{tail}'


async def test_register_starts_a_build_with_a_run_and_an_estimate(app, pg, client) -> None:
    alpha = project(app, pg, 'alpha')
    r = await client.post(url(alpha, '/spaces'), json=REGISTER)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body['space_key'] == 'mobileclip2_s2_v1'
    assert body['bytes_estimate'] > 0
    run = await client.get(url(alpha, f'/reindex-runs/{body["run_id"]}'))
    assert run.status_code == 200
    assert run.json()['job_id'] == body['job_id']
    assert run.json()['job_status'] == 'queued'
    listed = (await client.get(url(alpha, '/spaces'))).json()['spaces']
    assert [(s['space_key'], s['status'], s['role']) for s in listed] == [
        ('mobileclip2_s2_v1', 'pending', 'candidate')
    ]


async def test_a_second_register_for_one_space_is_409(app, pg, client) -> None:
    alpha = project(app, pg, 'alpha')
    assert (await client.post(url(alpha, '/spaces'), json=REGISTER)).status_code == 202
    again = await client.post(url(alpha, '/spaces'), json=REGISTER)
    assert again.status_code == 409
    assert again.json()['detail']['error'] == 'embedding_space_exists'


async def test_a_second_build_while_one_is_active_is_job_busy(app, pg, client) -> None:
    alpha = project(app, pg, 'alpha')
    assert (await client.post(url(alpha, '/spaces'), json=REGISTER)).status_code == 202
    again = await client.post(
        url(alpha, '/spaces/mobileclip2_s2_v1/build?target=item'), json={'mode': 'build'}
    )
    assert again.status_code == 409
    assert again.json()['detail']['error'] == 'job_busy'


async def test_a_repeat_with_the_same_idempotency_key_answers_the_first_job(
    app, pg, client
) -> None:
    alpha = project(app, pg, 'alpha')
    headers = {'Idempotency-Key': 'register-key-0001'}
    first = await client.post(url(alpha, '/spaces'), json=REGISTER, headers=headers)
    repeat = await client.post(url(alpha, '/spaces'), json=REGISTER, headers=headers)
    assert (first.status_code, repeat.status_code) == (202, 202)
    assert repeat.json()['job_id'] == first.json()['job_id']
    assert jobs_of(pg, alpha) == 1


async def test_a_data_path_without_twice_the_estimate_refuses_and_leaves_no_space(
    app, pg, client, capacity
) -> None:
    alpha = project(app, pg, 'alpha')
    capacity.docs, capacity.free = 1_000_000, 1_000_000_000  # needs ~2 GB x 2, has 1 GB
    r = await client.post(url(alpha, '/spaces'), json=REGISTER)
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'embedding_insufficient_storage'
    assert space_statuses(pg, alpha) == ['retired']
    assert jobs_of(pg, alpha) == 0


async def test_another_projects_spaces_and_runs_are_not_reachable(app, pg, client) -> None:
    alpha = project(app, pg, 'alpha')
    beta = project(app, pg, 'beta')
    started = (await client.post(url(alpha, '/spaces'), json=REGISTER)).json()
    assert (await client.get(url(beta, '/spaces'))).json() == {'spaces': []}
    run = await client.get(url(beta, f'/reindex-runs/{started["run_id"]}'))
    assert run.status_code == 404
    promote = await client.post(url(beta, '/spaces/mobileclip2_s2_v1/promote?target=item'), json={})
    assert promote.status_code == 404
    assert promote.json()['detail']['error'] == 'embedding_space_not_found'
