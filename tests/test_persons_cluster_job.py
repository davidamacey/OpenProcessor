"""POST /persons/cluster only queues the job: no grouping runs in the request."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fixtures.job_fixtures import job_env, kinds, producer, queue_hooks, wakes  # noqa: F401 - fixtures
from fixtures.job_rows import sql

from src.config.faces import require_faces
from src.core.db_session import install_database_error_handlers
from src.core.error_handlers import request_rejected_response
from src.core.exceptions import RequestRejectedError
from src.routers import persons
from src.services.face_identity import get_face_identity_service


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def client(pg, pg_env, job_env, producer, wakes, monkeypatch):  # noqa: F811 - fixtures
    from src.db.engine import dispose_engines

    sql(pg, "DELETE FROM jobs WHERE kind = 'persons_cluster'")  # the module's database is shared
    service = get_face_identity_service()

    async def refuse(*_a, **_k):
        raise AssertionError('the grouping ran inside the request')

    monkeypatch.setattr(service, 'auto_cluster_faces', refuse)
    app = FastAPI()
    app.dependency_overrides[require_faces] = lambda: None
    app.include_router(persons.router)
    install_database_error_handlers(app)
    app.add_exception_handler(RequestRejectedError, request_rejected_response)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://t') as c:
        yield c
    await dispose_engines()


async def test_the_route_queues_one_deployment_job_and_runs_no_grouping(
    client,
    pg,
    producer,  # noqa: F811 - the imported fixture
) -> None:
    reply = await client.post('/persons/cluster?similarity_threshold=0.6&min_faces_per_person=3')
    assert reply.status_code == 202
    job_id = reply.json()['job_id']
    [(kind, project, params)] = sql(
        pg,
        'SELECT kind, project_id, params FROM jobs WHERE id = %s',
        __import__('uuid').UUID(job_id),
    )
    assert (kind, project) == ('persons_cluster', None)
    assert params == {'similarity_threshold': 0.6, 'min_faces_per_person': 3}
    assert producer.calls[0] == ('send', job_id)


async def test_a_second_request_while_one_is_active_is_409(client) -> None:
    assert (await client.post('/persons/cluster')).status_code == 202
    again = await client.post('/persons/cluster')
    assert again.status_code == 409
    assert again.json()['detail']['error'] == 'job_busy'
