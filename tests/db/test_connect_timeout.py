"""A connect that stalls fails inside the bound with the typed outage error instead of hanging.

``10.255.255.1`` is not routable here: the SYN is dropped and the connect waits until the
driver's own ``connect_timeout`` (libpq, applied by psycopg to sync and async connects:
https://www.psycopg.org/psycopg3/docs/api/connections.html#psycopg.Connection.connect and the
SQLAlchemy psycopg dialect, which forwards ``connect_args`` to it:
https://docs.sqlalchemy.org/en/21/dialects/postgresql.html#module-sqlalchemy.dialects.postgresql.psycopg).
These tests need no database server; the bound is shortened to keep them quick.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import httpx
import psycopg
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from src.config.database import DatabaseSettings
from src.core.db_session import install_database_error_handlers
from src.db import engine as db_engine
from src.tasks import beat_lock


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.db

BOUND_S = 2
SLACK_S = 4


@pytest.fixture(autouse=True)
def _stalled_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for role in ('app', 'owner', 'super'):
        password = tmp_path / f'{role}_password'
        password.write_text('unused')
        monkeypatch.setenv(f'OP_DB_{role.upper()}_PASSWORD_FILE', str(password))
    monkeypatch.setenv('OP_DB_HOST', '10.255.255.1')
    monkeypatch.setenv('OP_DB_DIRECT_HOST', '10.255.255.1')
    monkeypatch.setenv('OP_DB_PORT', '5432')
    monkeypatch.setenv('WEB_CONCURRENCY', '1')
    monkeypatch.setattr(db_engine, 'CONNECT_TIMEOUT_S', BOUND_S)
    monkeypatch.setattr(beat_lock, 'CONNECT_TIMEOUT_S', BOUND_S)
    db_engine._engines.clear()  # test-only cache reset
    yield
    for engine in list(db_engine._engines.values()):
        (engine.sync_engine if hasattr(engine, 'sync_engine') else engine).dispose()
    db_engine._engines.clear()


def _within_the_bound(started: float) -> None:
    assert time.monotonic() - started < BOUND_S + SLACK_S


def test_the_poll_engine_gives_up_at_the_bound() -> None:
    started = time.monotonic()
    with pytest.raises(OperationalError) as caught, Session(db_engine.get_poll_engine()) as session:
        session.execute(text('SELECT 1'))
    assert isinstance(caught.value.orig, psycopg.errors.ConnectionTimeout)
    _within_the_bound(started)


def test_a_one_off_connection_gives_up_at_the_bound() -> None:
    started = time.monotonic()
    with pytest.raises(OperationalError), db_engine.one_off_connection():
        pass
    _within_the_bound(started)


def test_the_beat_lock_connection_gives_up_at_the_bound() -> None:
    started = time.monotonic()
    with pytest.raises(psycopg.errors.ConnectionTimeout):
        beat_lock.acquire(DatabaseSettings.from_env())
    _within_the_bound(started)


@pytest.mark.asyncio
async def test_a_request_on_the_api_engine_answers_503_database_unavailable_at_the_bound() -> None:
    app = FastAPI()
    install_database_error_handlers(app)
    factory = db_engine.current_async_engine()

    @app.get('/query')
    async def query() -> dict[str, int]:
        from sqlalchemy.ext.asyncio import AsyncSession

        async with AsyncSession(factory) as session:
            return {'one': (await session.execute(text('SELECT 1'))).scalar_one()}

    started = time.monotonic()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://t') as client:
        response = await client.get('/query')
    assert response.status_code == 503
    assert response.json()['code'] == 'database_unavailable'
    _within_the_bound(started)
