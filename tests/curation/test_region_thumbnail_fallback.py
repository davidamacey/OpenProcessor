"""``GET /crops/{id}/region_thumbnail`` falls back to the verifier-rejected
candidate box (DQ-B2 follow-up).

Before this, a ``verify_rejected`` item -- which never has
``region_bbox_norm``, only ``region_candidate_bbox_norm`` (see
``src/config/region_fields.py``) -- 404'd on this route even though the
item is still reviewable and reversible. This exercises the real FastAPI
route (not just the lower-level ``image_serving`` helpers already covered
by ``test_curation_images.py``), so the ``_fetch_crop`` source-includes
list and the router's fallback branch are both proven together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.config import CurationConfig, get_region_fields
from src.services.curation import image_serving


if TYPE_CHECKING:
    from pathlib import Path


F = get_region_fields()


class _FakeOSClient:
    """Minimal stand-in for the raw AsyncOpenSearch client ``_fetch_crop`` uses."""

    def __init__(self, docs: dict[str, dict[str, Any]]) -> None:
        self._docs = docs

    async def get(
        self,
        index: str,  # noqa: ARG002
        id: str,  # noqa: A002
        _source_includes: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            src = self._docs[id]
        except KeyError:
            raise Exception(f'NotFoundError: no such crop {id}') from None
        return {'_source': src}


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    img = Image.new('RGB', (640, 480), color=(40, 80, 120))
    path = tmp_path / 'source.jpg'
    img.save(path, format='JPEG', quality=88)
    return path


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_image: Path) -> TestClient:
    from src.routers import curation_images

    # Both resolve_crop_root and resolve_safe_path's absolute-path branch
    # read the config lazily via image_serving.get_curation_config() --
    # point that at tmp_path so an absolute image_path under it resolves.
    cfg = CurationConfig(source_root=tmp_path)
    monkeypatch.setattr(image_serving, 'get_curation_config', lambda: cfg)
    # Fresh, isolated cache so this test's assertions can't be polluted by
    # (or pollute) the process-wide THUMBNAIL_CACHE other tests share.
    monkeypatch.setattr(curation_images, 'THUMBNAIL_CACHE', image_serving.ThumbnailCache())

    docs = {
        'candidate_only': {
            'image_path': str(sample_image),
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            F.status: 'verify_rejected',
            F.candidate_bbox_norm: [0.1, 0.1, 0.4, 0.4],
        },
        'neither_box': {
            'image_path': str(sample_image),
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            F.status: 'no_region_visible',
        },
        'detected': {
            'image_path': str(sample_image),
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            F.status: 'detected',
            F.bbox_norm: [0.2, 0.2, 0.6, 0.6],
        },
    }
    fake_os = _FakeOSClient(docs)

    app = FastAPI()
    app.include_router(curation_images.crops_router)
    app.dependency_overrides[curation_images._raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


def test_region_thumbnail_renders_from_candidate_box(app_client: TestClient) -> None:
    r = app_client.get('/curation/crops/candidate_only/region_thumbnail')
    assert r.status_code == 200, r.text
    assert r.content.startswith(b'\xff\xd8'), 'must be JPEG magic bytes'


def test_region_thumbnail_404s_with_neither_box(app_client: TestClient) -> None:
    r = app_client.get('/curation/crops/neither_box/region_thumbnail')
    assert r.status_code == 404
    assert 'no region bbox' in r.json()['detail']


def test_region_thumbnail_still_renders_from_accepted_box(app_client: TestClient) -> None:
    """Unchanged behavior: an accepted region box still wins over any
    (nonexistent, here) candidate -- the fallback only kicks in when
    ``region_bbox_norm`` is absent."""
    r = app_client.get('/curation/crops/detected/region_thumbnail')
    assert r.status_code == 200, r.text
    assert r.content.startswith(b'\xff\xd8')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
