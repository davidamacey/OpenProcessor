"""Typed curation routes name their response model in the OpenAPI contract, and the models keep their wire keys."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


CONTRACT = Path(__file__).resolve().parents[1] / 'contracts' / 'openapi' / 'curation.json'
PREFIX = '/curation/projects/{project}'

# status None: the first 2xx response the operation declares.
TYPED = [
    ('get', '/models/status', None, 'ModelsStatusResponse'),
    ('get', '/clusters/representatives', None, 'ClusterRepresentativesPage'),
    ('post', '/probe/cancel', None, 'ProbeCancelResponse'),
    ('get', '/test_holdout/stats', None, 'HoldoutStats'),
    ('get', '/training_cohorts', None, 'TrainingCohortList'),
    ('get', '/class_sources', None, 'ClassSourceList'),
    ('post', '/classes', None, 'ClassCreated'),
    ('put', '/classes/{class_id}', None, 'RegistryClassEntry'),
    ('post', '/export/yolo', '200', 'ExportYoloResponse'),
    ('get', '/export/datasets', '200', 'ExportDatasetList'),
    ('post', '/export/single_class', '200', 'ExportSingleClassResponse'),
    ('get', '/export/single_class/status', '200', 'ExportSingleClassStatus'),
    ('get', '/stats/classes', '200', 'ClassStatsResponse'),
    ('get', '/train/manifest/{job_id}', '200', 'RunManifest'),
    ('post', '/prompt_packs/validate', '200', 'ValidationReport'),
    ('post', '/events/publish', '200', 'EventPublishAck'),
    ('get', '/events/stats', '200', 'EventStreamStats'),
    ('post', '/scores/compute', '202', 'ScoresState'),
    ('get', '/scores/status', '200', 'ScoresState'),
    ('post', '/scores/cancel', '200', 'ScoresCancelResponse'),
    ('get', '/scores/coverage', '200', 'ScoresCoverageResponse'),
    ('post', '/viz/projection/rebuild', '202', 'VizProjectionState'),
    ('get', '/viz/projection/status', '200', 'VizProjectionState'),
    ('post', '/viz/projection/cancel', '200', 'VizProjectionCancelResponse'),
    ('post', '/cluster/umap/rebuild', '200', 'UmapRebuildResponse'),
    ('get', '/regions/statuses', '200', 'RegionStatusCatalog'),
    ('get', '/regions/clusters', '200', 'RegionClusterList'),
    ('get', '/review/new_class_proposals/summary', '200', 'NewClassProposalSummary'),
    ('post', '/pipeline/auto_label', '200', 'AutoLabelRunSummary'),
]


@pytest.mark.parametrize(('method', 'path', 'status', 'model'), TYPED)
def test_response_schema_names_the_model(
    method: str, path: str, status: str | None, model: str
) -> None:
    doc = json.loads(CONTRACT.read_text())
    operation = doc['paths'][PREFIX + path][method]
    if status is None:
        status = next(s for s in operation['responses'] if s.startswith('2'))
    schema = operation['responses'][status]['content']['application/json']['schema']
    assert schema['$ref'].rsplit('/', 1)[1] == model
    assert doc['components']['schemas'][model].get('properties')


def test_holdout_stats_keeps_every_bucket_key() -> None:
    from src.schemas.curation.holdout import HoldoutStats

    body = {
        'total': 3,
        'by_class': [{'key': 2, 'doc_count': 3, 'deficient': True, 'extra_agg': 1}],
        'min_test_per_class': 20,
    }
    assert HoldoutStats(**body).model_dump() == body


def test_models_status_rows_keep_exactly_their_keys() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.schemas.curation.models_status import ModelsStatusResponse

    rows = [
        {
            'name': 'seg',
            'friendly_name': 'Segmenter',
            'role': 'r',
            'kind': 'external',
            'model_type': 't',
            'status': 'ready',
            'version': None,
            'last_error': None,
            'unloadable': False,
            'future_key': {'a': 1},
        },
        {
            'name': 'm',
            'friendly_name': 'M',
            'role': 'r',
            'kind': 'triton',
            'model_type': 't',
            'status': 'ready',
            'version': '1',
            'class_mapping': {'mapped_count': 1, 'unmapped': ['x']},
            'sharing_revision': None,
        },
    ]
    app = FastAPI()

    @app.get('/s', response_model=ModelsStatusResponse, response_model_exclude_unset=True)
    async def s() -> dict[str, object]:
        return {'models': rows}

    assert TestClient(app).get('/s').json() == {'models': rows}


def test_registry_class_entry_has_one_unprefixed_name() -> None:
    # Two models sharing a class name make FastAPI emit module-path names, which leak into TS types.
    names = json.loads(CONTRACT.read_text())['components']['schemas']
    assert 'RegistryClassEntry' in names
    assert not [n for n in names if n.endswith('__RegistryClassEntry')]


def test_class_merge_documents_dry_run_and_job() -> None:
    operation = json.loads(CONTRACT.read_text())['paths'][PREFIX + '/classes/merge']['post']
    refs = {
        code: operation['responses'][code]['content']['application/json']['schema']['$ref']
        for code in ('200', '202')
    }
    assert refs['200'].endswith('/ClassMergeDryRun')
    assert refs['202'].endswith('/JobView')


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


@pytest.mark.parametrize('path', ['/events', '/pipeline/events'])
def test_event_streams_document_their_frames(path: str) -> None:
    doc = json.loads(CONTRACT.read_text())
    ok = doc['paths'][PREFIX + path]['get']['responses']['200']
    assert 'text/event-stream' in ok['content']
    assert 'server-sent' in ok['description'].lower()


def _client(router_module: Any, *dependency_overrides: tuple[Any, Any]) -> Any:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(router_module.router)
    for dep, override in dependency_overrides:
        app.dependency_overrides[dep] = override
    return TestClient(app)


def test_region_status_catalog_body_is_kept_whole() -> None:
    from src.config.region_state import region_status_catalog
    from src.routers.curation import regions

    body = _client(regions).get('/regions/statuses').json()
    assert body == json.loads(json.dumps(region_status_catalog()))


def test_umap_no_residuals_branch_keeps_exactly_its_keys(monkeypatch) -> None:
    from src.core.dependencies import get_curation_opensearch
    from src.routers.curation import umap

    async def fake(_client: Any) -> dict[str, Any]:
        return {'status': 'no_residuals', 'n_residuals': 0, 'refit': False}

    monkeypatch.setattr(umap, 'umap_rebuild', fake)
    client = _client(umap, (get_curation_opensearch, lambda: object()))
    assert client.post('/cluster/umap/rebuild').json() == {
        'status': 'no_residuals',
        'n_residuals': 0,
        'refit': False,
    }


def test_events_stats_keeps_every_key(monkeypatch) -> None:
    from src.routers.curation import events

    stats = {
        'subscribers': 2,
        'clients_dropped': 1,
        'listener_running': True,
        'pubsub_connected': False,
    }

    class Listener:
        def stats(self, _project: str | None) -> dict[str, Any]:
            return stats

    monkeypatch.setattr(events, 'get_listener', lambda: Listener())
    monkeypatch.setattr(
        events, 'get_curation_config', lambda: type('Cfg', (), {'project_slug': 'p'})()
    )
    assert _client(events).get('/events/stats').json() == stats


def test_scores_state_keeps_job_keys() -> None:
    from src.routers.curation._latest_job_models import ScoresCancelResponse
    from src.services.curation.item_scores import job as scores_job

    state = {**scores_job.state_dict(None), 'cancelled': False}
    assert ScoresCancelResponse(**state).model_dump() == state
