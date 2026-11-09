"""With OP_FACES_ENABLED=false ingest never runs, indexes or reports faces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import test_ingest_batch_characterization as batch_char
import test_ingest_single_characterization as single_char
from fastapi.testclient import TestClient

from src.core.dependencies import get_visual_search_service


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def faces_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_FACES_ENABLED', 'false')


@pytest.mark.asyncio
async def test_single_ingest_skips_faces_even_when_asked(
    faces_off: None, monkeypatch: pytest.MonkeyPatch, pending_calls: Any
) -> None:
    got = await single_char.run_scenario(
        'full', monkeypatch, pending_calls
    )  # enable_faces defaults True
    assert got['face_calls'] == []
    assert [c[0] for c in got['opensearch']].count('index_face') == 0
    assert got['result']['num_faces'] == 0
    assert got['triton_calls'] == 1  # the rest of the pipeline still ran


@pytest.mark.asyncio
async def test_batch_ingest_skips_faces_even_when_asked(
    faces_off: None, monkeypatch: pytest.MonkeyPatch, pending_calls: Any
) -> None:
    got = await batch_char.run_scenario('full', monkeypatch, pending_calls)
    assert got['face'] == []
    assert 'bulk_index_faces' not in [c[0] for c in got['opensearch']]
    assert got['result']['processed'] == 4


class _FakeService:
    async def ingest_image(self, **_kw: object) -> dict[str, object]:
        return {'status': 'success', 'image_id': 'i1', 'num_faces': 0}

    async def ingest_batch(self, images_data: list[object], **_kw: object) -> dict[str, object]:
        return {'total': len(images_data), 'processed': len(images_data)}


@pytest.fixture
def client(faces_off: None) -> Iterator[TestClient]:
    from src.main import app

    app.dependency_overrides[get_visual_search_service] = _FakeService
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.pop(get_visual_search_service, None)


def test_ingest_response_reports_faces_off(client: TestClient) -> None:
    resp = client.post('/ingest', files={'image': ('a.jpg', b'x', 'image/jpeg')})
    assert resp.status_code == 201
    assert resp.json()['faces_enabled'] is False


def test_batch_ingest_response_reports_faces_off(client: TestClient) -> None:
    resp = client.post('/ingest/batch', files=[('images', ('a.jpg', b'x', 'image/jpeg'))])
    assert resp.status_code == 201
    assert resp.json()['faces_enabled'] is False
