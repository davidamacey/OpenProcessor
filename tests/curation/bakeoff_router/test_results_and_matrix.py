"""GET /bakeoff/results, /status, /runs and /matrix."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient


# =============================================================================
# GET /bakeoff/results, /status, /runs, /matrix
# =============================================================================


def _v2_comparison(job_id: str) -> dict[str, Any]:
    block = {
        'n_classes': 1,
        'map_50': 0.5,
        'map_50_95': 0.4,
        'map_75': 0.3,
        'ap_small': None,
        'ap_medium': None,
        'ap_large': 0.4,
        'precision': 1.0,
        'recall': 0.5,
        'f1': 0.667,
        'mean_iou': 0.8,
        'tp': 1,
        'fp': 0,
        'fn': 1,
    }
    common = {
        k: block[k]
        for k in ('n_classes', 'map_50', 'map_50_95', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn')
    }
    return {
        'schema_version': 2,
        'job_id': job_id,
        'profile': 'generic',
        'thresholds': {'conf_floor': 0.001, 'nms_iou': 0.7, 'op_conf': 0.25, 'op_iou': 0.45},
        'dataset': {
            'id': 'export:e1',
            'frozen_test_sha': 'f' * 16,
            'test_label_sha': 'a' * 16,
            'n_images': 2,
            'n_objects': 2,
            'n_background_images': 0,
        },
        'eval_classes': [{'eval_class_id': 0, 'name': 'a', 'n_gt': 2}],
        'common_classes': [0],
        'rank_by': 'map_50_95',
        'rank_scope': 'common',
        'models': [
            {
                'rank': 1,
                'model': 'run:r1',
                'display_name': 'r1',
                'source': 'run',
                'run_id': 'r1',
                'runtime': 'ultralytics',
                'imgsz': 640,
                'training_data': None,
                'overall': block,
                'common': common,
                'per_class': [
                    {
                        'eval_class_id': 0,
                        'name': 'a',
                        'n_gt': 2,
                        'covered': True,
                        'model_class_ids': [0],
                        'ap50': 0.5,
                        'ap50_95': 0.4,
                        'ap75': 0.3,
                        'precision': 1.0,
                        'recall': 0.5,
                        'f1': 0.667,
                        'tp': 1,
                        'fp': 0,
                        'fn': 1,
                    }
                ],
                'coverage': {
                    'n_eval_classes': 1,
                    'n_covered': 1,
                    'not_covered': [],
                    'unmapped_model_classes': [
                        {'model_class_id': 3, 'name': 'b', 'n_predictions': 4}
                    ],
                    'predictions_outside_scored_classes': 0,
                },
                'class_mapping': {'method': 'run_class_remap', 'warnings': []},
                'train_test_overlap': {'n_images': 0, 'fraction': 0.0},
                'latency_ms': {'mean': 1.0, 'p50': 1.0, 'p90': 1.0, 'p99': 1.0},
                'fps': 1000.0,
                'size_mb': 5.4,
                'per_stratum': {},
                'test_frames': 2,
            }
        ],
        'failed': [{'model': 'baseline:x', 'error': 'boom'}],
        'warnings': [],
        'n_models': 1,
    }


def _write_job_out(out: Path, job_id: str, *, status: dict[str, Any] | None = None) -> Path:
    job = out / job_id
    job.mkdir(parents=True)
    (job / 'status.json').write_text(
        json.dumps(
            status
            or {
                'schema_version': 2,
                'job_id': job_id,
                'state': 'done',
                'profile': 'generic',
                'datasets': ['export:e1', 'external:curated/x'],
                'models': ['run:r1'],
                'started_at': '2026-09-25T01:00:00+00:00',
                'finished_at': '2026-09-25T01:05:00+00:00',
                'progress': {'done': 2, 'total': 2},
                'completed': [{'dataset': 'export:e1', 'model': 'run:r1'}],
                'failed': [],
                'error': None,
            }
        )
    )
    return job


def test_results_serves_v2_and_rejects_v1(env: dict[str, Path], client: TestClient) -> None:
    job = _write_job_out(env['out'], 'j2')
    (job / 'export__e1').mkdir()
    (job / 'export__e1' / 'comparison.json').write_text(json.dumps(_v2_comparison('j2')))
    (job / 'external__curated__x').mkdir()
    (job / 'external__curated__x' / 'comparison.json').write_text(
        json.dumps({**_v2_comparison('j2'), 'dataset': {'id': 'external:curated/x'}})
    )

    r = client.get(
        '/curation/projects/default/bakeoff/results/j2'
    )  # defaults to the job's first dataset
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['dataset']['id'] == 'export:e1'
    row = body['models'][0]
    assert (row['rank'], row['common']['map_50_95'], row['coverage']['n_covered']) == (1, 0.4, 1)
    assert row['coverage']['unmapped_model_classes'][0]['n_predictions'] == 4
    r = client.get(
        '/curation/projects/default/bakeoff/results/j2', params={'dataset_id': 'external:curated/x'}
    )
    assert r.status_code == 200
    assert r.json()['dataset']['id'] == 'external:curated/x'
    assert (
        client.get(
            '/curation/projects/default/bakeoff/results/j2', params={'dataset_id': 'export:nope'}
        ).status_code
        == 404
    )

    v1 = _write_job_out(env['out'], 'j1')
    (v1 / 'export__e1').mkdir()
    (v1 / 'export__e1' / 'comparison.json').write_text(
        json.dumps({'models': [{'rank': 1, 'name': 'm', 'map_50': 0.5}], 'rank_by': 'map_50'})
    )
    r = client.get('/curation/projects/default/bakeoff/results/j1')
    assert r.status_code == 409
    assert r.json()['detail'] == (
        'bake-off result comparison.json has an unsupported schema (schema_version != 2)'
    )


def test_results_404_when_missing(env: dict[str, Path], client: TestClient) -> None:
    assert client.get('/curation/projects/default/bakeoff/results/no-such-job').status_code == 404
    assert client.get('/curation/projects/default/bakeoff/status/no-such-job').status_code == 404
    assert client.get('/curation/projects/default/bakeoff/matrix/no-such-job').status_code == 404


def test_runs_skip_v1_and_sort_newest_first(env: dict[str, Path], client: TestClient) -> None:
    assert client.get('/curation/projects/default/bakeoff/runs').json() == {'runs': []}
    _write_job_out(env['out'], 'older')
    newer = _write_job_out(env['out'], 'newer')
    st = json.loads((newer / 'status.json').read_text())
    st.update(job_id='newer', started_at='2026-09-26T00:00:00+00:00')
    (newer / 'status.json').write_text(json.dumps(st))
    legacy = env['out'] / 'legacy'
    legacy.mkdir()
    (legacy / 'status.json').write_text(json.dumps({'state': 'done', 'models': []}))
    runs = client.get('/curation/projects/default/bakeoff/runs').json()['runs']
    assert [r['job_id'] for r in runs] == ['newer', 'older']
    assert runs[0]['datasets'] == ['export:e1', 'external:curated/x']


def test_matrix_serves_v2(env: dict[str, Path], client: TestClient) -> None:
    job = _write_job_out(env['out'], 'jm')
    cell = {
        'map_50': 0.5,
        'map_50_95': 0.4,
        'precision': 1.0,
        'recall': 0.5,
        'f1': 0.667,
        'latency_ms': 1.0,
        'size_mb': 5.4,
        'coverage': 1.0,
        'rank': 1,
    }
    (job / 'matrix.json').write_text(
        json.dumps(
            {
                'schema_version': 2,
                'job_id': 'jm',
                'rank_by': 'map_50_95',
                'datasets': [
                    {
                        'id': 'export:e1',
                        'frozen_test_sha': None,
                        'test_label_sha': 'a',
                        'rank_scope': 'common',
                        'n_common_classes': 1,
                    }
                ],
                'models': [{'model': 'run:r1', 'display_name': 'r1', 'source': 'run'}],
                'metrics': ['map_50', 'map_50_95'],
                'cells': {'run:r1': {'export:e1': cell}},
                'best': {'export:e1': {'map_50_95': ['run:r1']}},
            }
        )
    )
    body = client.get('/curation/projects/default/bakeoff/matrix/jm').json()
    assert body['best'] == {'export:e1': {'map_50_95': ['run:r1']}}
    assert body['cells']['run:r1']['export:e1']['coverage'] == 1.0
    (job / 'matrix.json').write_text(json.dumps({'cells': {}, 'best': {'x': {'map_50': 'm'}}}))
    assert client.get('/curation/projects/default/bakeoff/matrix/jm').status_code == 409


# =============================================================================
