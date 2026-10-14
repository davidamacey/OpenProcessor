"""``/query/image/{id}``: only a missing document or index is "not found" / "nothing to
delete"; any other OpenSearch failure is an error response, not a clean answer."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError, NotFoundError

from src.core.dependencies import get_visual_search_service
from src.main import app


def _down() -> OpenSearchConnectionError:
    return OpenSearchConnectionError('N/A', 'opensearch is down', OSError('down'))


def _missing() -> NotFoundError:
    return NotFoundError(404, 'index_not_found_exception', {})


class _Client:
    def __init__(self, error: Exception, *, global_found: bool = True) -> None:
        self.error = error
        self.global_found = global_found

    async def get(self, **_: Any) -> dict[str, Any]:
        if not self.global_found:
            raise self.error
        return {'_source': {'image_id': 'img', 'image_path': '/x.jpg', 'indexed_at': '2026-01-01'}}

    async def count(self, **_: Any) -> dict[str, int]:
        raise self.error

    async def delete(self, **_: Any) -> None:
        raise self.error

    async def delete_by_query(self, **_: Any) -> dict[str, int]:
        raise self.error


@pytest.fixture
def serve():
    def _serve(client: _Client) -> TestClient:
        class _Service:
            class opensearch:  # noqa: N801 - attribute stand-in for the service's client
                pass

        _Service.opensearch.client = client
        app.dependency_overrides[get_visual_search_service] = _Service
        return TestClient(app)

    yield _serve
    app.dependency_overrides.pop(get_visual_search_service, None)


def test_an_unreadable_global_index_is_not_reported_as_not_found(serve) -> None:
    r = serve(_Client(_down(), global_found=False)).get('/query/image/img')
    assert r.status_code == 500


def test_a_missing_image_is_not_found(serve) -> None:
    r = serve(_Client(_missing(), global_found=False)).get('/query/image/img')
    assert r.status_code == 200
    assert r.json()['status'] == 'not_found'


def test_a_failing_detail_index_fails_the_request(serve) -> None:
    r = serve(_Client(_down())).get('/query/image/img')
    assert r.status_code == 500


def test_missing_detail_indexes_are_just_empty(serve) -> None:
    r = serve(_Client(_missing())).get('/query/image/img')
    assert r.status_code == 200
    assert r.json()['status'] == 'success'


def test_a_failed_delete_is_not_reported_as_deleted(serve) -> None:
    r = serve(_Client(_down())).delete('/query/image/img')
    assert r.status_code == 500


def test_deleting_what_is_not_there_is_not_an_error(serve) -> None:
    r = serve(_Client(_missing())).delete('/query/image/img')
    assert r.status_code == 200
