"""``GET /crops/{id}/region_thumbnail?box_id=`` renders one region box.

A rejected box (a ``verify_rejected`` item's only box) is still reviewable,
so it renders like any other state; the ``box_id`` is required and must
name a box the crop holds. This exercises the real FastAPI route (not just
the lower-level ``image_serving`` helpers already covered by
``test_curation_images.py``), so the ``_fetch_crop`` source-includes list
and the router's box lookup are proven together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.config import CurationConfig, get_region_fields
from src.services.curation import image_serving
from src.services.curation.region_boxes import RegionBox, boxes_write_fields


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
    # Two halves so boxes over different areas render differently.
    img.paste((220, 40, 40), (0, 0, 320, 480))
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
        'rejected_only': {
            'image_path': str(sample_image),
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            F.status: 'verify_rejected',
            **boxes_write_fields(
                [RegionBox(box_id='b1', bbox_norm=(0.1, 0.1, 0.4, 0.4), state='rejected')],
                current_src={},
            ),
        },
        'no_boxes': {
            'image_path': str(sample_image),
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            F.status: 'no_region_visible',
        },
        'detected': {
            'image_path': str(sample_image),
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            F.status: 'detected',
            **boxes_write_fields(
                [
                    RegionBox(box_id='b1', bbox_norm=(0.2, 0.2, 0.6, 0.6), state='accepted'),
                    RegionBox(box_id='b2', bbox_norm=(0.5, 0.5, 0.9, 0.9), state='accepted'),
                ],
                current_src={},
            ),
        },
    }
    fake_os = _FakeOSClient(docs)

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_images.crops_router)
    app.dependency_overrides[curation_images._raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


def _get(client: TestClient, crop_id: str, **params: str) -> Any:
    return client.get(f'/curation/projects/default/crops/{crop_id}/region_thumbnail', params=params)


def test_region_thumbnail_renders_a_rejected_box(app_client: TestClient) -> None:
    r = _get(app_client, 'rejected_only', box_id='b1')
    assert r.status_code == 200, r.text
    assert r.content.startswith(b'\xff\xd8'), 'must be JPEG magic bytes'


def test_region_thumbnail_renders_the_named_box_not_the_first(app_client: TestClient) -> None:
    """Each box renders from its own coordinates: two boxes of one crop
    produce two different images."""
    first = _get(app_client, 'detected', box_id='b1')
    second = _get(app_client, 'detected', box_id='b2')
    assert first.status_code == second.status_code == 200
    assert first.content != second.content


def test_region_thumbnail_requires_a_box_id(app_client: TestClient) -> None:
    r = _get(app_client, 'detected')
    assert r.status_code == 422
    assert r.json()['detail']['error'] == 'box_id_required'


def test_region_thumbnail_unknown_box_id_is_404(app_client: TestClient) -> None:
    r = _get(app_client, 'detected', box_id='b9')
    assert r.status_code == 404
    assert r.json()['detail']['error'] == 'unknown_box_id'


def test_region_thumbnail_404s_for_a_crop_with_no_boxes(app_client: TestClient) -> None:
    r = _get(app_client, 'no_boxes', box_id='b1')
    assert r.status_code == 404


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
