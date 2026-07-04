"""/metrics exposition + HTTP duration histogram behavior pins.

Verifies that a request through the real app records into
``http_request_duration_seconds`` with the matched route template, and
that ``/metrics`` renders a Prometheus exposition payload containing it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def client() -> TestClient:
    from src.main import app

    return TestClient(app)


def test_request_is_observed_with_route_template(client: TestClient) -> None:
    assert client.get('/live').status_code == 200
    body = client.get('/metrics').text
    assert 'http_request_duration_seconds' in body
    assert 'route="/live"' in body


def test_metrics_content_type_is_prometheus_exposition(client: TestClient) -> None:
    response = client.get('/metrics')
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/plain')


def test_unmatched_path_uses_low_cardinality_fallback(client: TestClient) -> None:
    assert client.get('/v1/nonexistent/abc123/deadbeef').status_code == 404
    body = client.get('/metrics').text
    # Only the first two path segments may appear — never the random tail.
    assert 'route="/v1/nonexistent"' in body
    assert 'abc123' not in body
