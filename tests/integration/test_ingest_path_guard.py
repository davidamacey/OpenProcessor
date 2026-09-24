"""Path-based ingest only accepts images the image server can later serve.

The image server serves an absolute ``image_path`` only from the configured
source roots (``OP_SOURCE_ROOT`` + ``OP_SOURCE_PATH_ALIASES``). An ingest that
stored a path outside them produced items whose thumbnails and source images
all failed with 400 -- discovered only after the index was populated. Such
paths must be refused at ingest instead.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from integration.ingest_fakes import FakeOpenSearch, FakeTritonPool, curation_app


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fastapi.testclient import TestClient


def _jpeg(tmp: Path, name: str, shade: int = 120) -> Path:
    buf = io.BytesIO()
    Image.new('RGB', (64, 48), (shade, 30, 30)).save(buf, format='JPEG')
    p = tmp / name
    p.write_bytes(buf.getvalue())
    return p


@pytest.fixture
def served_root(tmp_path: Path) -> Path:
    root = tmp_path / 'served'
    root.mkdir()
    return root


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, served_root: Path) -> Iterator[TestClient]:
    with curation_app(FakeOpenSearch(), FakeTritonPool(), monkeypatch) as c:
        from src.services.curation import image_serving

        monkeypatch.setattr(
            image_serving,
            '_configured_roots',
            lambda config=None: (served_root.resolve(),),  # noqa: ARG005
        )
        yield c


def test_batch_refuses_a_path_outside_the_source_roots(
    client: TestClient, tmp_path: Path, served_root: Path
) -> None:
    inside = _jpeg(served_root, 'in.jpg')
    outside = _jpeg(tmp_path, 'out.jpg', shade=200)
    resp = client.post(
        '/curation/ingest/batch',
        json={
            'items': [{'path': str(inside), 'source': 't'}, {'path': str(outside), 'source': 't'}]
        },
    )
    assert resp.status_code == 200, resp.text
    by_path: dict[str, Any] = {r['image_path']: r for r in resp.json()['results']}
    assert by_path[str(outside)]['status'] == 'failed'
    assert 'source root' in by_path[str(outside)]['error']
    assert by_path[str(inside)]['status'] != 'failed'


def test_single_ingest_refuses_a_path_outside_the_source_roots(
    client: TestClient, tmp_path: Path
) -> None:
    outside = _jpeg(tmp_path, 'single.jpg')
    resp = client.post('/curation/ingest/image', json={'path': str(outside), 'source': 't'})
    assert resp.status_code == 422, resp.text
    assert 'source root' in resp.json()['detail']
