"""GET /bakeoff/profiles and /bakeoff/baseline_models, and profile-scoped runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from curation.bakeoff_router._helpers import _live_export, _post_run


if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from fastapi.testclient import TestClient


# =============================================================================
# GET /bakeoff/profiles, /bakeoff/baseline_models
# =============================================================================


def test_profiles_lists_generic_only_by_default(env: dict[str, Path], client: TestClient) -> None:
    r = client.get('/curation/projects/default/bakeoff/profiles')
    assert r.status_code == 200
    body = r.json()
    assert [p['name'] for p in body['profiles']] == ['generic']
    [row] = body['profiles']
    assert (row['kind'], row['default'], row['class_filter']) == ('registered', True, [])
    assert row['rank_metric'] == 'map_50_95'
    assert row['context_class_ids'] == []
    assert 'target_class_id' not in row
    assert 'class_names' not in row
    assert (body['default_profile'], body['default_error'], body['count']) == ('generic', None, 1)


def test_profiles_configured_json_default_row(
    env: dict[str, Path], client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / 'widgets.json'
    f.write_text(json.dumps({'name': 'widgets', 'class_filter': ['widget']}))
    monkeypatch.setenv('OP_BAKEOFF_PROFILE', str(f))
    body = client.get('/curation/projects/default/bakeoff/profiles').json()
    [row] = [p for p in body['profiles'] if p['default']]
    assert (row['name'], row['kind'], row['class_filter']) == ('widgets', 'configured', ['widget'])
    assert body['count'] == len(body['profiles']) == 2

    monkeypatch.delenv('OP_BAKEOFF_PROFILE')
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_CONTEXT_CLASS_IDS', '4')
    body = client.get('/curation/projects/default/bakeoff/profiles').json()
    [row] = [p for p in body['profiles'] if p['default']]
    assert (row['name'], row['kind'], row['context_class_ids']) == ('generic', 'registered', [4])
    # No registry configured in this env -- falls back to the id's string
    # form rather than 500ing or guessing a name.
    assert row['context_class_names'] == ['4']


def test_profiles_context_class_names_resolved_by_registry_id(
    env: dict[str, Path], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``context_class_names`` is paired with ``context_class_ids`` by id
    lookup against the live registry, same order/length."""
    from src.routers.curation import bakeoff

    class _Cls:
        def __init__(self, class_id: int, class_name: str) -> None:
            self.class_id = class_id
            self.class_name = class_name

    class _Loaded:
        def __init__(self, classes: list[_Cls]) -> None:
            self.classes = classes

    class _FakeRegistry:
        def load(self) -> _Loaded:
            return _Loaded([_Cls(4, 'guardrail'), _Cls(7, 'cone')])

    monkeypatch.setattr(bakeoff, 'get_class_registry', _FakeRegistry)
    monkeypatch.setenv('OP_BAKEOFF_PROFILE_CONTEXT_CLASS_IDS', '4,7,99')
    body = client.get('/curation/projects/default/bakeoff/profiles').json()
    [row] = [p for p in body['profiles'] if p['default']]
    assert row['context_class_ids'] == [4, 7, 99]
    # id 99 has no registry entry -- falls back to its string form, never
    # a guess -- while 4 and 7 resolve by id, not by list position.
    assert row['context_class_names'] == ['guardrail', 'cone', '99']


def test_profiles_bad_default_is_reported(
    env: dict[str, Path], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_BAKEOFF_PROFILE', 'no_such_profile')
    body = client.get('/curation/projects/default/bakeoff/profiles').json()
    assert body['default_profile'] is None
    assert 'unknown bake-off profile' in body['default_error']
    assert not any(p['default'] for p in body['profiles'])


def test_run_rejects_unknown_profile(env: dict[str, Path], client: TestClient) -> None:
    _live_export(env)
    r = _post_run(
        client,
        {
            'profile': 'no_such_profile',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'custom', 'name': 'c', 'backend': 'ultralytics'}],
        },
    )
    assert r.status_code == 400
    assert 'unknown bake-off profile' in r.json()['detail']


def test_run_profile_class_filter_narrows_scored_classes(
    env: dict[str, Path], client: TestClient, tmp_path: Path
) -> None:
    _live_export(env)
    prof = tmp_path / 'two.json'
    prof.write_text(json.dumps({'name': 'two', 'class_filter': ['Miata', 'vw']}))
    r = _post_run(
        client,
        {
            'job_id': 'jp',
            'profile': str(prof),
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'custom', 'name': 'c', 'backend': 'ultralytics'}],
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()['profile'] == 'two'
    assert r.json()['datasets'][0]['n_eval_classes'] == 2
    spec = json.loads((env['jobs'] / 'jp.job.json').read_text())
    assert spec['profile'] == str(prof)
    assert spec['datasets'][0]['eval_class_ids'] == [37, 78]


def test_default_baseline_registry_is_empty(env: dict[str, Path], client: TestClient) -> None:
    body = client.get('/curation/projects/default/bakeoff/baseline_models').json()
    assert body == {'baselines': [], 'count': 0}


def test_baseline_models_per_profile_and_run_lookup(
    env: dict[str, Path], client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import bakeoff_resolve

    reg = tmp_path / 'baselines.json'
    reg.write_text(
        json.dumps(
            {
                'baselines': [
                    {
                        'name': 'coco-yolo11n',
                        'backend': 'ultralytics',
                        'weights': './weights/yolo11n.pt',
                        'imgsz': 640,
                        'training_data': 'COCO',
                    },
                    {'name': 'bad entry!', 'backend': 'ultralytics'},
                ]
            }
        )
    )
    monkeypatch.setattr(bakeoff_resolve, 'BASELINES_PATH', reg)
    body = client.get('/curation/projects/default/bakeoff/baseline_models').json()
    assert body['count'] == 1
    [b] = body['baselines']
    assert b == {
        'name': 'coco-yolo11n',
        'backend': 'ultralytics',
        'weights': './weights/yolo11n.pt',
        'imgsz': 640,
        'mode': 'full',
        'class_map': None,
        'backend_options': {},
        'training_data': 'COCO',
        'triton_model': None,
    }
    for bad in ('license_plate', 'nope', '../../etc/x.json'):
        r = client.get(
            '/curation/projects/default/bakeoff/baseline_models', params={'profile': bad}
        )
        assert r.status_code == 400, bad
    assert (
        client.get(
            '/curation/projects/default/bakeoff/baseline_models', params={'profile': 'generic'}
        ).json()['count']
        == 1
    )

    _live_export(env)
    r = _post_run(
        client,
        {
            'job_id': 'jb',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'baseline', 'name': 'coco-yolo11n'}],
        },
    )
    assert r.status_code == 200, r.text
    [m] = json.loads((env['jobs'] / 'jb.job.json').read_text())['models']
    assert (m['model'], m['weights'], m['training_data']) == (
        'baseline:coco-yolo11n',
        './weights/yolo11n.pt',
        'COCO',
    )
