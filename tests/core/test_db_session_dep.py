"""DbSessionDep: one session per request, closed before the response, and a failed commit is a 5xx."""

from __future__ import annotations

import asyncio
import typing

import httpx
import psycopg
import pytest
from fastapi import Depends, FastAPI
from sqlalchemy import text

from src.core.db_session import DbSessionDep
from src.db.session import deployment_tx


if typing.TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


def test_the_dependency_is_declared_with_function_scope() -> None:
    _, dependency = typing.get_args(DbSessionDep)
    assert dependency.scope == 'function'


@pytest.fixture
def scratch(pg_env: PgDatabase):
    """A table whose commit fails with a serialization error, to force the failure at COMMIT."""
    with psycopg.connect(pg_env.conninfo('super'), autocommit=True) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS scratch_commit (id int)')
        conn.execute('GRANT ALL ON scratch_commit TO op_app')
        conn.execute(
            'CREATE OR REPLACE FUNCTION scratch_commit_fail() RETURNS trigger LANGUAGE plpgsql AS $$ '
            "BEGIN RAISE EXCEPTION 'forced' USING ERRCODE = '40001'; END $$"
        )
        conn.execute('DROP TRIGGER IF EXISTS scratch_commit_fail ON scratch_commit')
        conn.execute(
            'CREATE CONSTRAINT TRIGGER scratch_commit_fail AFTER INSERT ON scratch_commit '
            'DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION scratch_commit_fail()'
        )
    yield pg_env
    with psycopg.connect(pg_env.conninfo('super'), autocommit=True) as conn:
        conn.execute('DROP TABLE IF EXISTS scratch_commit')
        conn.execute('DROP FUNCTION IF EXISTS scratch_commit_fail()')


def _client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url='http://t')


@pytest.mark.asyncio
async def test_a_commit_that_fails_is_a_5xx_not_a_2xx(scratch) -> None:
    app = FastAPI()

    @app.post('/write')
    async def write(session: DbSessionDep) -> dict[str, str]:
        async with deployment_tx(session, 'test'):
            await session.execute(text('INSERT INTO scratch_commit VALUES (1)'))
        return {'stored': 'yes'}

    async with _client(app) as client:
        response = await client.post('/write')
    assert response.status_code == 500
    assert 'stored' not in response.text


@pytest.mark.asyncio
async def test_a_normal_request_commits(scratch) -> None:
    app = FastAPI()

    @app.post('/write')
    async def write(session: DbSessionDep) -> dict[str, str]:
        async with deployment_tx(session, 'test'):
            await session.execute(text("INSERT INTO deployment_counters (name) VALUES ('web')"))
        return {'stored': 'yes'}

    @app.get('/read')
    async def read(session: DbSessionDep) -> dict[str, int]:
        async with deployment_tx(session, 'test'):
            n = (
                await session.execute(
                    text("SELECT count(*) FROM deployment_counters WHERE name = 'web'")
                )
            ).scalar_one()
        return {'n': n}

    async with _client(app) as client:
        assert (await client.post('/write')).json() == {'stored': 'yes'}
        assert (await client.get('/read')).json() == {'n': 1}


@pytest.mark.asyncio
async def test_an_exception_in_the_handler_rolls_back_and_surfaces(scratch) -> None:
    app = FastAPI()

    @app.post('/fail')
    async def fail(session: DbSessionDep) -> None:
        async with deployment_tx(session, 'test'):
            await session.execute(text("INSERT INTO deployment_counters (name) VALUES ('ghost')"))
            raise RuntimeError('handler failed')

    @app.get('/count')
    async def count(session: DbSessionDep) -> dict[str, int]:
        async with deployment_tx(session, 'test'):
            n = (
                await session.execute(
                    text("SELECT count(*) FROM deployment_counters WHERE name = 'ghost'")
                )
            ).scalar_one()
        return {'n': n}

    async with _client(app) as client:
        assert (await client.post('/fail')).status_code == 500
        assert (await client.get('/count')).json() == {'n': 0}


@pytest.mark.asyncio
async def test_two_concurrent_requests_get_two_sessions(scratch) -> None:
    app = FastAPI()
    arrived = asyncio.Event()
    release = asyncio.Event()
    seen: list[int] = []

    @app.get('/session')
    async def session_id(session: DbSessionDep) -> dict[str, int]:
        seen.append(id(session))
        if len(seen) == 2:
            arrived.set()
        await asyncio.wait_for(arrived.wait(), 5)
        await release.wait()
        return {'id': id(session)}

    async with _client(app) as client:
        first = asyncio.create_task(client.get('/session'))
        second = asyncio.create_task(client.get('/session'))
        await asyncio.wait_for(arrived.wait(), 5)
        release.set()
        ids = {(await first).json()['id'], (await second).json()['id']}
    assert len(ids) == 2


_EVENTS: list[str] = []


async def _failing_teardown() -> typing.AsyncIterator[None]:
    try:
        yield
    finally:
        _EVENTS.append('teardown')
        raise RuntimeError('teardown failed')


_Scoped = typing.Annotated[None, Depends(_failing_teardown, scope='function')]


@pytest.mark.asyncio
async def test_function_scope_runs_the_teardown_before_the_response_is_sent() -> None:
    """The reason for scope="function": a teardown failure must still be able to change the status."""
    _EVENTS.clear()
    app = FastAPI()

    @app.get('/x')
    async def route(_dep: _Scoped) -> dict[str, str]:
        return {'ok': 'yes'}

    async with _client(app) as client:
        response = await client.get('/x')
    assert _EVENTS == ['teardown']
    assert response.status_code == 500
