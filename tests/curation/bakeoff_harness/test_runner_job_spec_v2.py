"""Profile propagation and v2 job-spec handling in ``bakeoff_runner``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from curation.bakeoff_harness._helpers import _BOX, _FakeDetector, _mc_dataset
from scripts.curation.bakeoff import bakeoff_runner, run
from scripts.curation.bakeoff.backends import registry
from scripts.curation.bakeoff.freeze import test_sha as label_sha


if TYPE_CHECKING:
    from pathlib import Path


# --- bakeoff_runner.py (job spec v2) ------------------------------------------------------


def _v2_dataset(tmp_path: Path) -> tuple[Path, dict]:
    root = _mc_dataset(
        tmp_path / 'ds',
        {'a': [(0, *_BOX)], 'b': [(1, *_BOX)], 'bg': []},
        ['dog', 'cat'],
        fill={'a': 10, 'b': 200, 'bg': 100},
    )
    entry = {
        'id': 'export:ds',
        'dir_name': 'export__ds',
        'path': str(root),
        'test_label_sha': label_sha(root)[0],
        'frozen_test_sha': '0' * 16,
        'eval_class_ids': [0, 1],
    }
    return root, entry


def _v2_model(key: str, class_map_by_dataset: dict | None = None, **kw) -> dict:
    return {
        'model': key,
        'display_name': kw.pop('display_name', key),
        'source': 'custom',
        'run_id': None,
        'backend': kw.pop('backend', 'ultralytics'),
        'weights': None,
        'imgsz': 64,
        'mode': 'full',
        'backend_options': {},
        'triton_model': None,
        'training_data': None,
        'class_map_by_dataset': class_map_by_dataset or {'export:ds': None},
        'train_test_overlap_by_dataset': {'export:ds': None},
        **kw,
    }


def _v2_spec(tmp_path: Path, datasets_: list[dict], models: list[dict], **kw) -> dict:
    return {
        'schema_version': 2,
        'job_id': 'j2',
        'profile': 'generic',
        'out_dir': str(tmp_path / 'out'),
        'datasets': datasets_,
        'models': models,
        'quantize': None,
        **kw,
    }


def test_runner_rejects_changed_test_split(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(bakeoff_runner, '_run_task', lambda *a: calls.append(a))
    _, entry = _v2_dataset(tmp_path)
    actual = entry['test_label_sha']
    entry['test_label_sha'] = 'deadbeefdeadbeef'
    status = bakeoff_runner.run_job(_v2_spec(tmp_path, [entry], [_v2_model('custom:m')]))
    assert calls == []
    assert {
        'stage': None,
        'dataset': 'export:ds',
        'model': None,
        'error': f'test split changed since enqueue: expected deadbeefdeadbeef, now {actual}',
    } in status['failed']
    assert status['state'] == 'error'
    on_disk = json.loads((tmp_path / 'out/status.json').read_text())
    assert on_disk['schema_version'] == 2


def test_runner_rejects_v1_job_spec(tmp_path: Path) -> None:
    status = bakeoff_runner.run_job(
        {'job_id': 'old', 'out_dir': str(tmp_path / 'out'), 'dataset': '/d', 'models': []}
    )
    assert status['state'] == 'error'
    assert 'schema_version' in status['error']


def test_runner_rejects_unknown_profile(tmp_path: Path) -> None:
    _, entry = _v2_dataset(tmp_path)
    status = bakeoff_runner.run_job(_v2_spec(tmp_path, [entry], [], profile='nope'))
    assert status['state'] == 'error'
    assert 'unknown bake-off profile' in status['error']
    assert json.loads((tmp_path / 'out/status.json').read_text())['state'] == 'error'


def test_runner_passes_v2_model_fields_to_run(tmp_path: Path, monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_task(ds_path, ds_out, ds_id, model, gpu):
        ds_out.mkdir(parents=True, exist_ok=True)
        seen.append(bakeoff_runner._model_argv(ds_path, ds_out, ds_id, model))
        return model['model'], True, None

    monkeypatch.setattr(bakeoff_runner, '_run_task', fake_task)
    _, entry = _v2_dataset(tmp_path)
    model = _v2_model(
        'run:r1',
        {'export:ds': {'0': 1}},
        backend_options={'providers': 'CPUExecutionProvider'},
        train_test_overlap_by_dataset={'export:ds': {'n_images': 1, 'fraction': 0.5}},
        mode='both',
    )
    prof = tmp_path / 'prof.json'
    prof.write_text(json.dumps({'name': 'mine', 'context_class_ids': [4]}))
    status = bakeoff_runner.run_job(_v2_spec(tmp_path, [entry], [model], profile=str(prof)))
    assert status['state'] == 'done'
    assert status['models'] == ['run:r1:full', 'run:r1:crop']
    argv = seen[0]

    def flag(name: str) -> str:
        return argv[argv.index(name) + 1]

    assert flag('--profile') == str(prof)
    assert json.loads(flag('--class-map-json')) == {'0': 1}
    assert json.loads(flag('--backend-options-json')) == {'providers': 'CPUExecutionProvider'}
    assert json.loads(flag('--train-test-overlap-json')) == {'n_images': 1, 'fraction': 0.5}
    assert {a[a.index('--model-key') + 1] for a in seen} == {'run:r1:full', 'run:r1:crop'}
    assert flag('--source') == 'custom'
    assert '--gt-class-id' not in argv


def test_runner_v2_end_to_end_with_fake_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(registry, '_REGISTRY', dict(registry._REGISTRY))  # undo after the test
    registry.register_backend('fake-scripted', lambda args: _FakeDetector(args.name))

    def in_process_task(ds_path, ds_out, ds_id, model, gpu):
        argv = bakeoff_runner._model_argv(ds_path, ds_out, ds_id, model)
        assert argv[1:3] == ['-m', 'scripts.curation.bakeoff.run']
        assert run.main([*argv[3:], '--no-mlflow']) == 0
        return model['model'], True, None

    monkeypatch.setattr(bakeoff_runner, '_run_task', in_process_task)
    _, entry = _v2_dataset(tmp_path)
    models = [
        _v2_model('custom:by-name', display_name='by name', backend='fake-scripted'),
        _v2_model(
            'custom:dog-only',
            {'export:ds': {'1': 0}},
            display_name='dog only',
            backend='fake-scripted',
        ),
    ]
    status = bakeoff_runner.run_job(_v2_spec(tmp_path, [entry], models))
    out = tmp_path / 'out'
    assert status['state'] == 'done', status
    assert json.loads((out / 'status.json').read_text())['schema_version'] == 2
    assert not (out / 'comparison.json').exists()  # per-dataset files only
    ds_out = out / 'export__ds'
    report = json.loads((ds_out / 'custom_by-name.json').read_text())
    assert report['schema_version'] == 2
    assert report['class_mapping']['method'] == 'names'
    assert report['overall']['map_50'] == pytest.approx(1.0)
    assert (report['test_frames'], report['background_frames']) == (3, 1)
    comp = json.loads((ds_out / 'comparison.json').read_text())
    assert comp['schema_version'] == 2
    assert comp['dataset']['id'] == 'export:ds'
    assert comp['common_classes'] == [0]
    assert [m['model'] for m in comp['models']] == ['custom:by-name', 'custom:dog-only']
    dog_only = comp['models'][1]
    assert dog_only['coverage']['not_covered'] == [{'eval_class_id': 1, 'name': 'cat'}]
    assert dog_only['class_mapping']['method'] == 'explicit'
    matrix = json.loads((out / 'matrix.json').read_text())
    assert matrix['schema_version'] == 2
    assert matrix['datasets'][0]['id'] == 'export:ds'
    assert sorted(matrix['best']['export:ds']['map_50_95']) == ['custom:by-name', 'custom:dog-only']
