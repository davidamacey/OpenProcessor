"""DQ-M9: exports refuse to be empty, and preflight blocks empty or stale exports.

- ``POST /export/yolo`` with nothing exportable (0 validated items, or
  every validated item on an unregistered class) is a ``422`` with the
  reason, and writes no directory and no ``current`` symlink.
- Every export stamps the items index it was built from (``items_index``:
  name, uuid, creation time) into its manifest.
- ``POST /train/preflight`` blocks an export with 0 images, and an export
  whose stamp names a different items index (the index was rebuilt), or —
  for an unstamped export — whose ``exported_at`` predates the current
  index's creation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.clients.curation_opensearch import ClassRegistry
from src.config import CurationConfig
from src.services.curation.export import GenericYoloExportService
from src.services.curation.export_readiness import (
    NothingToExportError,
    export_generation_check,
    export_size_check,
)


if TYPE_CHECKING:
    from pathlib import Path


INDEX_UUID = 'uuid-current'
INDEX_CREATED = datetime(2026, 9, 24, 13, 50, tzinfo=UTC)


def _settings(uuid: str = INDEX_UUID, created: datetime = INDEX_CREATED) -> dict[str, Any]:
    return {
        'op_items': {
            'settings': {
                'index': {'uuid': uuid, 'creation_date': str(int(created.timestamp() * 1000))}
            }
        }
    }


class _Indices:
    def __init__(self, settings: dict[str, Any]) -> None:
        self._settings = settings

    async def get_settings(self, *, index: str) -> dict[str, Any]:  # noqa: ARG002
        return self._settings


class _FakeOpenSearch:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs
        self.indices = _Indices(_settings())

    async def search(self, index: str, body: dict[str, Any], scroll: str | None = None) -> dict:  # noqa: ARG002
        hits = [{'_id': d['crop_id'], '_source': d} for d in self._docs]
        return {'_scroll_id': 's', 'hits': {'hits': hits}}

    async def scroll(self, scroll_id: str, scroll: str) -> dict:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, scroll_id: str) -> None:  # noqa: ARG002
        return None


def _service(tmp_path: Path, docs: list[dict[str, Any]]) -> GenericYoloExportService:
    cfg = CurationConfig(export_root=tmp_path / 'exports')
    reg = ClassRegistry(path=tmp_path / 'registry' / 'class_registry.json')
    reg.add_class('sedan')
    return GenericYoloExportService(_FakeOpenSearch(docs), config=cfg, registry=reg)


def _doc(crop_id: str, class_id: int) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_id': f'img-{crop_id}',
        'image_path': f'/data/{crop_id}.jpg',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'class_id': class_id,
        'class_name': 'x',
    }


# ----------------------------------------------------------------- export side


@pytest.mark.asyncio
async def test_export_with_no_validated_items_refuses(tmp_path: Path) -> None:
    service = _service(tmp_path, [])
    with pytest.raises(NothingToExportError, match='0 items'):
        await service.export_dataset(copy_images=False)
    root = tmp_path / 'exports'
    assert not root.exists() or not any(root.iterdir())


@pytest.mark.asyncio
async def test_export_with_only_unregistered_classes_refuses(tmp_path: Path) -> None:
    service = _service(tmp_path, [_doc('a', 77), _doc('b', 78)])
    with pytest.raises(NothingToExportError, match='not in the class registry'):
        await service.export_dataset(copy_images=False)
    assert not (tmp_path / 'exports' / 'current').exists()


@pytest.mark.asyncio
async def test_export_manifest_records_the_items_index_generation(tmp_path: Path) -> None:
    service = _service(tmp_path, [_doc('a', 0)])
    result = await service.export_dataset(copy_images=False)
    manifest = json.loads(open(result.manifest_path).read())  # noqa: SIM115
    assert manifest['items_index']['uuid'] == INDEX_UUID
    assert manifest['items_index']['created_at'] == INDEX_CREATED.isoformat()


def test_export_router_maps_nothing_to_export_to_422(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    async def _refuse(self: Any, **_kw: Any) -> Any:
        raise NothingToExportError('0 items are class_validated')

    monkeypatch.setattr(GenericYoloExportService, 'export_dataset', _refuse)
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: AsyncMock()
    r = TestClient(app).post('/curation/export/yolo', json={})
    assert r.status_code == 422
    assert '0 items are class_validated' in r.json()['detail']


# ------------------------------------------------------------ preflight rules


CURRENT = {'index': 'op_items', 'uuid': INDEX_UUID, 'created_at': INDEX_CREATED.isoformat()}


def test_generation_same_uuid_is_ok() -> None:
    sev, _msg, _d = export_generation_check({'items_index': {'uuid': INDEX_UUID}}, CURRENT)
    assert sev == 'ok'


def test_generation_other_uuid_blocks_even_if_exported_later() -> None:
    manifest = {'items_index': {'uuid': 'uuid-old'}, 'exported_at': '2030-01-01T00:00:00+00:00'}
    sev, msg, _d = export_generation_check(manifest, CURRENT)
    assert sev == 'block'
    assert 'rebuilt' in msg


def test_unstamped_export_before_index_creation_blocks() -> None:
    manifest = {'exported_at': '2026-09-24T03:20:32+00:00'}
    assert export_generation_check(manifest, CURRENT)[0] == 'block'
    assert export_generation_check({'exported_at': '2026-09-24T14:00:00Z'}, CURRENT)[0] == 'ok'


def test_generation_unknown_without_evidence() -> None:
    assert export_generation_check({'items_index': {'uuid': 'x'}}, None)[0] == 'unknown'
    assert export_generation_check({}, CURRENT)[0] == 'unknown'


def test_size_check() -> None:
    assert export_size_check({'image_count': 0})[0] == 'block'
    assert export_size_check({'split_counts': {'train': 0, 'val': 0, 'test': 0}})[0] == 'block'
    assert export_size_check({'image_count': 23})[0] == 'ok'
    assert export_size_check({})[0] == 'unknown'


# ------------------------------------------------------- preflight end-to-end


@pytest.fixture
def train_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path / 'jobs'))
    from src.services.training import gpu_arbiter

    monkeypatch.setattr(gpu_arbiter, '_state_dir', lambda: tmp_path)

    class _Reg:
        def load(self) -> Any:
            class _Snap:
                classes: list[Any] = []

            return _Snap()

        def get(self, _cid: int) -> Any:
            return None

    monkeypatch.setattr('src.routers.curation_train.get_class_registry', lambda: _Reg())
    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'hits': {'hits': [], 'total': {'value': 0}}})
    fake.count = AsyncMock(return_value={'count': 0})
    fake.indices.get_settings = AsyncMock(return_value=_settings())

    from src.routers.curation._common import _raw_opensearch_dep
    from src.routers.curation_train import router as train_router

    app = FastAPI()
    app.include_router(train_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _single_class_export(tmp_path: Path, **manifest: Any) -> str:
    d = tmp_path / 'single'
    d.mkdir()
    base = {
        'dataset_kind': 'single_class',
        'class_name': 'region',
        'positive_images': 20,
        'image_count': 23,
        'split_counts': {'train': 19, 'val': 2, 'test': 2},
    }
    (d / 'manifest.json').write_text(json.dumps({**base, **manifest}))
    return str(d)


def _check(body: dict[str, Any], name: str) -> dict[str, Any]:
    return next(c for c in body['checks'] if c['name'] == name)


def test_preflight_blocks_single_class_export_older_than_the_index(
    train_client: TestClient, tmp_path: Path
) -> None:
    export_dir = _single_class_export(tmp_path, exported_at='2026-09-24T03:20:32+00:00')
    r = train_client.post(
        '/curation/train/preflight', json={'dataset_export_dir': export_dir, 'profile': 'medium'}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert _check(body, 'export_generation')['severity'] == 'block'
    assert body['blocked'] is True


def test_preflight_passes_generation_for_a_current_stamp(
    train_client: TestClient, tmp_path: Path
) -> None:
    export_dir = _single_class_export(tmp_path, items_index=CURRENT)
    body = train_client.post(
        '/curation/train/preflight', json={'dataset_export_dir': export_dir, 'profile': 'medium'}
    ).json()
    assert _check(body, 'export_generation')['severity'] == 'ok'
    assert _check(body, 'export_not_empty')['severity'] == 'ok'


def test_preflight_blocks_an_empty_multi_class_export(
    train_client: TestClient, tmp_path: Path
) -> None:
    d = tmp_path / 'multi'
    (d / 'labels' / 'train').mkdir(parents=True)
    (d / 'manifest.json').write_text(
        json.dumps({'image_count': 0, 'split_counts': {}, 'items_index': CURRENT})
    )
    body = train_client.post(
        '/curation/train/preflight', json={'dataset_export_dir': str(d), 'profile': 'medium'}
    ).json()
    assert _check(body, 'export_not_empty')['severity'] == 'block'
    assert body['blocked'] is True


@pytest.mark.asyncio
async def test_single_class_export_with_no_matching_item_refuses(tmp_path: Path) -> None:
    from src.config.region_fields import RegionFields
    from src.services.curation.export_single_class import (
        SingleClassExportProfile,
        SingleClassExportService,
    )

    cfg = CurationConfig(export_root=tmp_path / 'exports', source_root=tmp_path / 'images')
    reg = ClassRegistry(path=tmp_path / 'registry' / 'class_registry.json')
    reg.add_class('car')
    service = SingleClassExportService(
        _FakeOpenSearch([]),
        profile=SingleClassExportProfile(name='cars', class_ids=(0,)),
        config=cfg,
        registry=reg,
        region_fields=RegionFields(),
    )
    with pytest.raises(NothingToExportError, match='nothing to export'):
        await service.export(copy_images=False)
    root = tmp_path / 'exports'
    assert not root.exists() or not any(p.is_file() for p in root.rglob('*'))
