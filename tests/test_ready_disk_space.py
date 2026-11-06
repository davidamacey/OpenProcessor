"""Low disk space: /ready reports it, critical refuses new writes with 507.

shutil.disk_usage: https://docs.python.org/3/library/shutil.html#shutil.disk_usage
FastAPI dependencies in decorators: https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/
"""

from __future__ import annotations

import shutil
from collections import namedtuple
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from src.core import disk_space
from src.core.error_handlers import request_rejected_response
from src.core.exceptions import RequestRejectedError


if TYPE_CHECKING:
    from collections.abc import Callable

Usage = namedtuple('Usage', 'total used free')  # mirrors shutil's result type

GUARDED = [
    ('src.routers.ingest.single', '', 'POST'),
    ('src.routers.ingest.batch', '/batch', 'POST'),
    ('src.routers.ingest.directory', '/directory', 'POST'),
    ('src.routers.curation.ingest', '/ingest/image', 'POST'),
    ('src.routers.curation.ingest', '/ingest/batch', 'POST'),
    ('src.routers.curation.ingest_upload', '/ingest/upload', 'POST'),
    ('src.routers.curation.ingest_bulk', '/ingest/bulk', 'POST'),
    ('src.routers.curation.ingest_objects', '/ingest/objects', 'POST'),
    ('src.routers.curation.ingest_sources', '/ingest/sources/{source_id}/run', 'POST'),
    ('src.routers.curation.datasets', '/datasets/uploads', 'POST'),
    ('src.routers.curation.datasets', '/datasets/imports', 'POST'),
    ('src.routers.models_export', '/upload', 'POST'),
]


def _free(monkeypatch: pytest.MonkeyPatch, percent: float) -> None:
    monkeypatch.setattr(
        shutil, 'disk_usage', lambda _path: Usage(1000, 1000 - int(percent * 10), int(percent * 10))
    )


@pytest.fixture
def health_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers import health

    async def up_http(_url: str) -> tuple[bool, str]:
        return True, 'HTTP 200'

    async def up_triton() -> tuple[bool, str]:
        return True, 'ok'

    async def up_models() -> tuple[bool, str, list[dict[str, str]]]:
        return True, 'ok', []

    async def no_opensearch_disk(_url: str) -> float | None:
        return None

    table: dict[str, Callable[[], None]] = {'database': lambda: None, 'queue': lambda: None}
    monkeypatch.setattr(health, 'startup_probes', lambda _settings: table)
    monkeypatch.setattr(health, '_probe_http', up_http)
    monkeypatch.setattr(health, '_probe_triton', up_triton)
    monkeypatch.setattr(health, '_probe_required_models', up_models)
    monkeypatch.setattr(health, '_opensearch_free_ratio', no_opensearch_disk)
    app = FastAPI()
    app.include_router(health.router)
    return TestClient(app)


def test_fifty_percent_free_is_ok(
    health_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _free(monkeypatch, 50)
    response = health_client.get('/ready')
    assert response.status_code == 200
    disk = response.json()['services']['disk']
    assert disk['status'] == 'ok'
    assert disk['volumes']['data']['free_ratio'] == 0.5


def test_five_percent_free_is_low_but_still_ready(
    health_client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _free(monkeypatch, 5)
    with caplog.at_level('WARNING'):
        response = health_client.get('/ready')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ready'
    assert body['services']['disk']['status'] == 'low'
    assert 'volume data' in caplog.text


def test_two_percent_free_is_503_naming_disk(
    health_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _free(monkeypatch, 2)
    response = health_client.get('/ready')
    assert response.status_code == 503
    disk = response.json()['services']['disk']
    assert disk['status'] == 'critical'
    assert disk['ok'] is False


def test_thresholds_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_DISK_LOW_PCT', '20')
    monkeypatch.setenv('OP_DISK_CRITICAL_PCT', '8')
    assert disk_space.classify(0.15) == 'low'
    assert disk_space.classify(0.05) == 'critical'
    assert disk_space.classify(0.5) == 'ok'


def _guarded_app() -> FastAPI:
    app = FastAPI()
    app.add_exception_handler(RequestRejectedError, request_rejected_response)  # type: ignore[arg-type]

    @app.post('/start', dependencies=[disk_space.DISK_GUARD])
    async def start() -> dict[str, str]:
        return {'started': 'yes'}

    return app


def test_a_start_is_refused_with_507_at_critical(monkeypatch: pytest.MonkeyPatch) -> None:
    _free(monkeypatch, 2)
    response = TestClient(_guarded_app()).post('/start')
    assert response.status_code == 507
    assert response.json()['detail']['error'] == 'insufficient_storage'


def test_a_start_is_accepted_at_fifty_percent(monkeypatch: pytest.MonkeyPatch) -> None:
    _free(monkeypatch, 50)
    assert TestClient(_guarded_app()).post('/start').status_code == 200


def test_a_start_is_accepted_when_only_low(monkeypatch: pytest.MonkeyPatch) -> None:
    _free(monkeypatch, 5)
    assert TestClient(_guarded_app()).post('/start').status_code == 200


@pytest.mark.parametrize(('module', 'path', 'method'), GUARDED)
def test_every_start_route_carries_the_guard(module: str, path: str, method: str) -> None:
    import importlib

    router = importlib.import_module(module).router

    def matches(r: object) -> bool:
        if not isinstance(r, APIRoute) or method not in r.methods:
            return False
        return r.path == router.prefix if path == '' else r.path.endswith(path)

    routes = [r for r in router.routes if matches(r)]
    assert len(routes) == 1, f'{method} {path} in {module}: {len(routes)} routes'
    route = routes[0]
    assert any(d.dependency is disk_space.require_disk_space for d in route.dependencies), (
        f'{module} {path} lacks the disk guard'
    )
