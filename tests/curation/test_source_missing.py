"""Image routes answer a typed ``source_missing`` when the source is unreachable (#405).

A bare ``404 image missing`` left the frontend unable to say *why* a tile was
blank. The answer now carries ``error: source_missing``, the root's alias name
and whether the root itself is mounted, and never a host path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import CurationConfig
from src.services.curation import image_serving


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class _FakeOSClient:
    def __init__(self, docs: dict[str, dict[str, Any]]) -> None:
        self.docs = docs

    async def get(
        self,
        index: str,  # noqa: ARG002
        id: str,  # noqa: A002
        _source_includes: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            return {'_source': self.docs[id]}
        except KeyError:
            raise Exception(f'NotFoundError: no such crop {id}') from None


def _client(
    monkeypatch: pytest.MonkeyPatch, cfg: CurationConfig, docs: dict[str, dict[str, Any]]
) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import images as curation_images

    monkeypatch.setattr(image_serving, 'get_curation_config', lambda: cfg)
    monkeypatch.setattr(curation_images, 'THUMBNAIL_CACHE', image_serving.ThumbnailCache())
    app = FastAPI()
    mount_curation_routers(app, curation_images.crops_router, curation_images.router)
    app.dependency_overrides[curation_images._raw_opensearch_dep] = lambda: _FakeOSClient(docs)
    return TestClient(app)


def test_missing_file_under_mounted_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = CurationConfig(source_root=tmp_path)
    docs = {'gone': {'image_path': str(tmp_path / 'x.jpg'), 'bbox_norm': [0, 0, 1, 1]}}
    client = _client(monkeypatch, cfg, docs)
    for route in ('thumbnail', 'image'):
        r = client.get(f'/curation/projects/default/crops/gone/{route}')
        assert r.status_code == 404, r.text
        detail = r.json()['detail']
        assert detail['error'] == 'source_missing'
        assert detail['alias'] == 'source_root'
        assert detail['root_mounted'] is True


def test_unmounted_alias_root_is_named(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gone_root = tmp_path / 'not_mounted'
    cfg = CurationConfig(source_root=tmp_path, source_path_aliases={'archive': gone_root})
    docs = {'off': {'image_path': str(gone_root / 'a.jpg'), 'bbox_norm': [0, 0, 1, 1]}}
    client = _client(monkeypatch, cfg, docs)
    r = client.get('/curation/projects/default/crops/off/thumbnail')
    assert r.status_code == 404, r.text
    detail = r.json()['detail']
    assert detail['error'] == 'source_missing'
    assert detail['alias'] == 'archive'
    assert detail['root_mounted'] is False
    assert str(tmp_path) not in detail['message']


def test_serve_route_reports_source_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _client(monkeypatch, CurationConfig(source_root=tmp_path), {})
    r = client.get('/curation/projects/default/images/serve', params={'path': 'nope.jpg'})
    assert r.status_code == 404, r.text
    assert r.json()['detail']['error'] == 'source_missing'
