"""Routes typed in lane typed-responses-2 name their response model in the OpenAPI contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


CONTRACT = Path(__file__).resolve().parents[1] / 'contracts' / 'openapi' / 'curation.json'
PREFIX = '/curation/projects/{project}'

TYPED = [
    ('post', '/export/yolo', 'ExportYoloResponse'),
    ('get', '/export/datasets', 'ExportDatasetList'),
    ('post', '/export/single_class', 'ExportSingleClassResponse'),
    ('get', '/export/single_class/status', 'ExportSingleClassStatus'),
    ('get', '/stats/classes', 'ClassStatsResponse'),
    ('get', '/train/manifest/{job_id}', 'RunManifest'),
    ('post', '/prompt_packs/validate', 'ValidationReport'),
]


@pytest.mark.parametrize(('method', 'path', 'model'), TYPED)
def test_response_schema_names_the_model(method: str, path: str, model: str) -> None:
    doc = json.loads(CONTRACT.read_text())
    operation = doc['paths'][PREFIX + path][method]
    schema = operation['responses']['200']['content']['application/json']['schema']
    assert schema['$ref'].rsplit('/', 1)[1] == model
    assert doc['components']['schemas'][model].get('properties')


def _client_for(router_module: Any) -> Any:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(router_module.router)
    return TestClient(app)


def test_single_class_status_branches_keep_exactly_their_keys(tmp_path: Path, monkeypatch) -> None:
    from src.routers.curation import export_single_class as mod

    client = _client_for(mod)
    monkeypatch.setattr(
        mod, '_resolve_current_dir', lambda name: (_ for _ in ()).throw(FileNotFoundError())
    )
    assert client.get('/export/single_class/status').json() == {
        'status': 'idle',
        'last_run': None,
        'profile_name': 'single_class',
    }
    monkeypatch.setattr(mod, '_resolve_current_dir', lambda name: tmp_path)
    assert client.get('/export/single_class/status').json() == {
        'status': 'unknown',
        'last_run': None,
        'profile_name': 'single_class',
        'export_dir': str(tmp_path),
    }
    (tmp_path / mod.MANIFEST_FILENAME).write_text(
        json.dumps({'dataset_sha': 'abc', 'finished_at': 't', 'split_counts': {'train': 1}})
    )
    body = client.get('/export/single_class/status').json()
    assert body['dataset_sha'] == 'abc'
    assert body['split_counts'] == {'train': 1}
    assert body['class_names'] is None
    assert len(body) == 15


def test_class_stats_keeps_raw_aggregations() -> None:
    from src.routers.curation._stats_models import ClassStatsResponse

    body = {
        'by_class': {'buckets': [{'key': 1, 'doc_count': 2}]},
        'by_source': {'x': 1},
        'classes': [
            {
                'class_id': 1,
                'class_name': 'car',
                'count': 2,
                'validated_count': 1,
                'trainable': 1,
                'trainable_gap': 0,
                'adequacy': 'ok',
                'aug_target': 5,
                'aug_gap': 4,
                'test_count': 3,
                'test_deficient': True,
            }
        ],
        'sort_applied': 'class_id:asc',
        'sort_options': [{'id': 'class_id:asc', 'label': 'ID (asc)'}],
        'thresholds': {
            'block_below': 1,
            'warn_below': 2,
            'min_test_per_class': 3,
            'min_train_per_class': 4,
            'min_val_per_class': 5,
            'aug_target_min': 6,
            'aug_target_max': 7,
        },
    }
    assert ClassStatsResponse(**body).model_dump(exclude_unset=True) == body


def test_export_yolo_response_keeps_every_key() -> None:
    from src.routers.curation.export import ExportYoloResponse

    body = {
        'status': 'success',
        'export_dir': '/x',
        'version_tag': '',
        'manifest_path': '/x/m',
        'dataset_sha': 's',
        'image_count': 1,
        'object_count': 2,
        'split_counts': {'train': 1, 'val': 0, 'test': 0},
        'class_count': 3,
        'classes_with_objects': 1,
        'split_object_counts': {'train': 2, 'val': 0, 'test': 0},
        'require_fully_labeled_images': False,
        'unlabeled_items_on_exported_images': 0,
        'images_with_unlabeled_items': 0,
        'images_dropped_not_fully_labeled': 0,
        'skipped_items': {'no_image_id': 0, 'no_usable_box_or_class': 0},
        'collapse_near_duplicates': False,
        'started_at': None,
        'finished_at': 't',
    }
    assert ExportYoloResponse(**body).model_dump() == body
