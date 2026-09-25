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
    assert by_path[str(outside)]['error_kind'] == 'unservable_path'
    assert by_path[str(inside)]['status'] != 'failed'


def test_batch_refuses_a_label_txt_path_outside_the_source_roots(
    client: TestClient, tmp_path: Path, served_root: Path
) -> None:
    """BA-5: label_txt_path gets the same root guard as the image path --
    a client-controlled label file path must not escape the configured
    source roots either."""
    inside_image = _jpeg(served_root, 'labeled.jpg')
    outside_label = tmp_path / 'evil.txt'
    outside_label.write_text('0 0.5 0.5 0.2 0.2\n')
    resp = client.post(
        '/curation/ingest/batch',
        json={
            'items': [
                {
                    'path': str(inside_image),
                    'source': 't',
                    'label_txt_path': str(outside_label),
                }
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()['results'][0]
    assert result['status'] == 'failed'
    assert result['error_kind'] == 'unservable_path'
    assert str(outside_label) in result['error']


def test_batch_over_the_configured_item_cap_is_413(
    client: TestClient, served_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BA-5: /ingest/batch enforces the served item cap."""
    import src.config.curation as curation_config_mod

    monkeypatch.setenv('OP_BATCH_MAX_ITEMS_PER_REQUEST', '1')
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    a = _jpeg(served_root, 'a.jpg')
    b = _jpeg(served_root, 'b.jpg')
    resp = client.post(
        '/curation/ingest/batch',
        json={'items': [{'path': str(a), 'source': 't'}, {'path': str(b), 'source': 't'}]},
    )
    assert resp.status_code == 413


def test_single_ingest_refuses_a_path_outside_the_source_roots(
    client: TestClient, tmp_path: Path
) -> None:
    outside = _jpeg(tmp_path, 'single.jpg')
    resp = client.post('/curation/ingest/image', json={'path': str(outside), 'source': 't'})
    assert resp.status_code == 422, resp.text
    assert 'source root' in resp.json()['detail']


def test_batch_rejects_a_malformed_body_instead_of_a_silent_noop(
    client: TestClient, served_root: Path
) -> None:
    """F-22: {'paths': [...]} previously validated against IngestBatchRequest
    with `items` defaulting to [], returning 200 status=success with
    all-zero counts -- a silent no-op indistinguishable from ingesting an
    empty batch on purpose. The wrong key must now 422."""
    inside = _jpeg(served_root, 'wrongkey.jpg')
    resp = client.post(
        '/curation/ingest/batch',
        json={'paths': [str(inside)]},
    )
    assert resp.status_code == 422, resp.text


def test_batch_rejects_an_empty_items_list(client: TestClient) -> None:
    """F-22: an empty (or omitted) items list is also a no-op that should
    422 rather than silently returning success with zero results."""
    resp = client.post('/curation/ingest/batch', json={'items': []})
    assert resp.status_code == 422, resp.text

    resp = client.post('/curation/ingest/batch', json={})
    assert resp.status_code == 422, resp.text


def test_single_ingest_rejects_a_malformed_body(client: TestClient, served_root: Path) -> None:
    """F-22: same silent-ignore class of bug on the single-image route --
    an unknown key like 'image_path' instead of 'path' must 422."""
    inside = _jpeg(served_root, 'single_wrongkey.jpg')
    resp = client.post('/curation/ingest/image', json={'image_path': str(inside), 'source': 't'})
    assert resp.status_code == 422, resp.text
