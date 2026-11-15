"""An ingest request the frame engine could not take answers 503 with ``Retry-After``: the bulk
sender retries exactly that, and a person's upload sees a busy answer instead of a 500."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.app.factory import create_app
from src.core.exceptions import EngineUnavailableError


def test_the_engine_being_unavailable_is_a_503_with_retry_after() -> None:
    application = create_app()

    @application.get('/__engine_down')
    def down() -> None:
        raise EngineUnavailableError('engine not ready')

    response = TestClient(application, raise_server_exceptions=False).get('/__engine_down')

    assert response.status_code == 503
    assert response.headers['Retry-After'] == '5'
    assert response.json()['code'] == 'engine_unavailable'
