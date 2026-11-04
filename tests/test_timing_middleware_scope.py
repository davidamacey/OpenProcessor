"""The timing middleware injects ``total_time_ms`` and ``request_id`` into core
inference JSON bodies only; curation has its own serializer and wire names."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.app.factory import create_app


@pytest.fixture(scope='module')
def client() -> TestClient:
    application = create_app()

    @application.get('/curation/probe/ingest')
    async def _curation() -> dict[str, int]:
        return {'a': 1}

    @application.get('/detect/probe')
    async def _core() -> dict[str, int]:
        return {'a': 1}

    @application.get('/v1/detect/probe')
    async def _core_v1() -> dict[str, int]:
        return {'a': 1}

    return TestClient(application, raise_server_exceptions=False)


def test_curation_body_is_not_rewritten(client: TestClient) -> None:
    r = client.get('/curation/probe/ingest')

    assert r.json() == {'a': 1}
    assert 'X-Process-Time' in r.headers


@pytest.mark.parametrize('path', ['/detect/probe', '/v1/detect/probe'])
def test_core_inference_body_keeps_timing_keys(client: TestClient, path: str) -> None:
    body = client.get(path).json()

    assert body['a'] == 1
    assert 'total_time_ms' in body
    assert 'request_id' in body
