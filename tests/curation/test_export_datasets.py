"""``GET /export/datasets`` lists multi-class AND single-class exports with a
``kind`` + ``profile_name`` per row and honors ``?kind=`` / ``?profile_name=``
(contract audit S4)."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.config
from src.config import get_curation_config
from src.routers.curation._common import config as curation_config


if TYPE_CHECKING:
    from pathlib import Path


P = curation_config.api_prefix


def _write_export(d: Path, **meta: Any) -> None:
    d.mkdir(parents=True)
    (d / 'manifest.json').write_text(json.dumps(meta), encoding='utf-8')


@pytest.fixture
def export_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / 'exports'
    # Multi-class versions directly under the root, `current` -> v2.
    _write_export(root / 'v1', version_tag='v1', exported_at='2026-01-01', image_count=10)
    _write_export(
        root / 'v2', version_tag='v2', exported_at='2026-02-01', image_count=20, object_count=31
    )
    (root / 'current').symlink_to(root / 'v2')
    # Two single-class profiles, each with its own `current`.
    _write_export(
        root / 'regions_only' / 'r1',
        dataset_kind='single_class',
        version_tag='r1',
        exported_at='2026-03-01',
        image_count=5,
        class_count=1,
    )
    _write_export(
        root / 'regions_only' / 'r2',
        dataset_kind='single_class',
        version_tag='r2',
        exported_at='2026-04-01',
        image_count=6,
        class_count=1,
    )
    (root / 'regions_only' / 'current').symlink_to(root / 'regions_only' / 'r1')
    _write_export(
        root / 'subset_a' / 's1',
        dataset_kind='single_class',
        version_tag='s1',
        exported_at='2026-05-01',
    )
    # Not a dataset at all: ignored.
    (root / 'scratch' / 'junk').mkdir(parents=True)

    cfg = dataclasses.replace(get_curation_config(), export_root=root)
    monkeypatch.setattr(src.config, 'get_curation_config', lambda: cfg)
    return root


@pytest.fixture
def client(export_root: Path) -> Any:
    from src.routers.curation import router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    with TestClient(app) as c:
        yield c


def _rows(client: TestClient, **params: Any) -> list[dict[str, Any]]:
    r = client.get(f'{P}/export/datasets', params=params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['count'] == len(body['datasets'])
    return body['datasets']


def test_lists_both_kinds_newest_first(client: TestClient) -> None:
    rows = _rows(client)
    assert [r['version_tag'] for r in rows] == ['s1', 'r2', 'r1', 'v2', 'v1']
    for row in rows:
        assert set(row) == {
            'kind',
            'profile_name',
            'export_dir',
            'version_tag',
            'image_count',
            'object_count',
            'split_counts',
            'dataset_sha',
            'exported_at',
            'class_count',
            'is_current',
        }


def test_kind_and_profile_per_row(client: TestClient) -> None:
    by_tag = {r['version_tag']: r for r in _rows(client)}
    assert by_tag['v1']['kind'] == 'yolo'
    # Multi-class rows count images and objects separately; an export made
    # before object counts were recorded reports null.
    assert (by_tag['v2']['image_count'], by_tag['v2']['object_count']) == (20, 31)
    assert by_tag['v1']['object_count'] is None
    assert by_tag['v1']['profile_name'] is None
    assert by_tag['r1']['kind'] == 'single_class'
    assert by_tag['r1']['profile_name'] == 'regions_only'
    assert by_tag['s1']['profile_name'] == 'subset_a'


def test_is_current_is_per_profile(client: TestClient) -> None:
    current = {r['version_tag'] for r in _rows(client) if r['is_current']}
    # Multi-class current is v2; regions_only's own current is r1;
    # subset_a has no current symlink.
    assert current == {'v2', 'r1'}


def test_kind_filter(client: TestClient) -> None:
    assert {r['version_tag'] for r in _rows(client, kind='yolo')} == {'v1', 'v2'}
    assert {r['version_tag'] for r in _rows(client, kind='single_class')} == {'r1', 'r2', 's1'}
    assert _rows(client, kind='something_else') == []


def test_profile_name_filter(client: TestClient) -> None:
    assert {r['version_tag'] for r in _rows(client, profile_name='regions_only')} == {'r1', 'r2'}
    assert _rows(client, kind='yolo', profile_name='regions_only') == []


def test_export_kinds_match_the_methods_export_axis() -> None:
    """``kind`` values are the same ids /methods advertises on the export axis."""
    from src.routers.curation.export import MULTI_CLASS_DATASET_KIND, SINGLE_CLASS_DATASET_KIND
    from src.services.curation.strategy_registry import _export_strategies

    export_ids = {s['id'] for s in _export_strategies()}
    assert {MULTI_CLASS_DATASET_KIND, SINGLE_CLASS_DATASET_KIND} <= export_ids
