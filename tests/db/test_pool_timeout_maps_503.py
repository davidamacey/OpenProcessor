"""An exhausted API pool answers 503 database_busy within the pool timeout; a dead database 503 database_unavailable."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text

from src.core.db_session import install_database_error_handlers
from src.db.engine import POOL_TIMEOUT_S
from src.db.session import async_session_factory


pytestmark = pytest.mark.db


def _app() -> FastAPI:
    app = FastAPI()
    install_database_error_handlers(app)
    factory = async_session_factory()

    @app.get('/hold')
    async def hold() -> dict[str, str]:
        async with factory() as session:
            await session.execute(text('SELECT 1'))
            await asyncio.sleep(POOL_TIMEOUT_S + 3)
        return {'held': 'yes'}

    @app.get('/query')
    async def query() -> dict[str, int]:
        async with factory() as session:
            return {'one': (await session.execute(text('SELECT 1'))).scalar_one()}

    return app


@pytest.mark.asyncio
async def test_an_exhausted_pool_answers_503_database_busy_within_the_timeout(
    pg_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_DB_API_CONNECTIONS_TOTAL', '1')
    monkeypatch.setenv('WEB_CONCURRENCY', '1')
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url='http://t') as client:
        holder = asyncio.create_task(client.get('/hold'))
        await asyncio.sleep(0.5)  # let the holder take the only connection
        started = time.monotonic()
        response = await client.get('/query')
        waited = time.monotonic() - started
        assert response.status_code == 503
        assert response.json()['code'] == 'database_busy'
        assert response.headers['retry-after'] == '2'
        assert POOL_TIMEOUT_S - 1 <= waited <= POOL_TIMEOUT_S + 2
        assert (await holder).status_code == 200
        # The pool recovers once the connection is returned.
        assert (await client.get('/query')).json() == {'one': 1}


@pytest.mark.asyncio
async def test_an_unreachable_database_answers_503_database_unavailable(
    pg_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_DB_HOST', '127.0.0.1')
    monkeypatch.setenv('OP_DB_DIRECT_HOST', '127.0.0.1')
    monkeypatch.setenv('OP_DB_PORT', '9')
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url='http://t') as client:
        response = await client.get('/query')
    assert response.status_code == 503
    assert response.json()['code'] == 'database_unavailable'
    assert response.headers['retry-after'] == '5'


@pytest.mark.asyncio
async def test_a_cancelled_statement_is_not_reported_as_an_outage(pg_env) -> None:
    app = _app()
    factory = async_session_factory()

    @app.get('/slow')
    async def slow() -> dict[str, int]:
        async with factory() as session:
            await session.execute(text("SET LOCAL statement_timeout = '100ms'"))
            await session.execute(text('SELECT pg_sleep(2)'))
        return {}

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url='http://t') as client:
        response = await client.get('/slow')
    assert response.status_code == 500
