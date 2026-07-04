"""X-Request-ID correlation behavior pins.

A client-supplied ``X-Request-ID`` must be echoed back and bound to the
logging context for the duration of the request; when absent, the
middleware must generate one.

Uses the real application object so the middleware wiring itself is under
test; TestClient is used without its context manager so the lifespan
(Triton pool, OpenSearch warm-up) never runs.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def client() -> TestClient:
    from src.main import app

    return TestClient(app)


def test_client_supplied_request_id_is_echoed(client: TestClient) -> None:
    response = client.get('/live', headers={'X-Request-ID': 'test-corr-123'})
    assert response.status_code == 200
    assert response.headers['X-Request-ID'] == 'test-corr-123'


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    response = client.get('/live')
    assert response.status_code == 200
    generated = response.headers.get('X-Request-ID')
    assert generated
    assert generated != '-'


def test_bind_request_id_updates_context() -> None:
    from src.core.logging import bind_request_id, clear_request_id, get_request_id

    assert get_request_id() == '-'
    bind_request_id('ctx-abc')
    try:
        assert get_request_id() == 'ctx-abc'
    finally:
        clear_request_id()
    assert get_request_id() == '-'
