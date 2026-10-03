"""#83: project delete cleans the MLflow experiment over REST and reports
an explicit outcome (the API image has no mlflow module)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from src.services.projects.mlflow_cleanup import delete_experiment


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _run(name: str, handler):
    async def go():
        async with _client(handler) as http:
            return await delete_experiment(name, http=http)

    return asyncio.run(go())


def test_not_configured_is_reported(monkeypatch) -> None:
    monkeypatch.delenv('MLFLOW_TRACKING_URI', raising=False)
    got = asyncio.run(delete_experiment('openprocessor-x'))
    assert got.outcome == 'skipped_not_configured'
    assert got.reason


def test_deletes_found_experiment(monkeypatch) -> None:
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000/')
    seen: list[tuple[str, str, bytes]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path, req.content))
        if req.method == 'GET':
            assert req.url.params['experiment_name'] == 'openprocessor-x'
            return httpx.Response(200, json={'experiment': {'experiment_id': '7'}})
        return httpx.Response(200, json={})

    got = _run('openprocessor-x', handler)
    assert got.to_wire() == {'outcome': 'done', 'reason': None}
    assert seen[1][1] == '/api/2.0/mlflow/experiments/delete'
    assert b'"7"' in seen[1][2]


def test_missing_experiment_is_done(monkeypatch) -> None:
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    got = _run('n', lambda _r: httpx.Response(404, json={'error_code': 'RESOURCE_DOES_NOT_EXIST'}))
    assert got.outcome == 'done'


@pytest.mark.parametrize('status', [500, 503])
def test_server_error_is_failed_with_reason(monkeypatch, status: int) -> None:
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    got = _run('n', lambda _r: httpx.Response(status))
    assert got.outcome == 'failed'
    assert str(status) in (got.reason or '')


def test_unreachable_is_failed(monkeypatch) -> None:
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')

    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('refused')

    got = _run('n', boom)
    assert got.outcome == 'failed'
    assert 'ConnectError' in (got.reason or '')
