"""/ready probes the database and the queue and names every dependency that is down;
/live probes nothing (FastAPI testing: https://fastapi.tiangolo.com/tutorial/testing/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.db.schema_check import ProbeFailed


if TYPE_CHECKING:
    from collections.abc import Callable


def _ok() -> None:
    return None


def _down(reason: str) -> Callable[[], None]:
    def probe() -> None:
        raise ProbeFailed(reason)

    return probe


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers import health

    async def up_http(url: str) -> tuple[bool, str]:
        return True, 'HTTP 200'

    async def up_triton() -> tuple[bool, str]:
        return True, 'is_server_live OK'

    async def up_models() -> tuple[bool, str, list[dict[str, str]]]:
        return True, 'models READY', []

    monkeypatch.setattr(health, '_probe_http', up_http)
    monkeypatch.setattr(health, '_probe_triton', up_triton)
    monkeypatch.setattr(health, '_probe_required_models', up_models)
    app = FastAPI()
    app.include_router(health.router)
    return TestClient(app)


def _probes(monkeypatch: pytest.MonkeyPatch, **probes: Callable[[], None]) -> None:
    from src.routers import health

    table = {'database': _ok, 'queue': _ok, **probes}
    monkeypatch.setattr(health, 'startup_probes', lambda _settings: table)


def test_both_up_is_ready(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _probes(monkeypatch)
    response = client.get('/ready')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ready'
    assert body['services']['database']['ok'] is True
    assert body['services']['queue']['ok'] is True


def test_database_down_is_503_naming_database_while_live_answers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _probes(monkeypatch, database=_down('OperationalError'))
    response = client.get('/ready')
    assert response.status_code == 503
    assert response.json()['services']['database'] == {
        'ok': False,
        'detail': 'OperationalError',
    }
    assert client.get('/live').status_code == 200
    assert client.get('/health').status_code == 503


def test_every_down_dependency_is_named(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _probes(monkeypatch, database=_down('db'), queue=_down('broker did not answer'))
    body = client.get('/ready').json()
    assert body['services']['database']['ok'] is False
    assert body['services']['queue']['ok'] is False
    assert body['services']['queue']['detail'] == 'broker did not answer'
