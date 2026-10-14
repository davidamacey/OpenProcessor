"""POST /bakeoff/run: request validation, model/class mapping and the queued job it writes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from curation.bakeoff_router._helpers import (
    LIVE_REMAP,
    _live_export,
    _post_run,
    make_export,
    make_run,
)
from scripts.curation.bakeoff.freeze import test_sha as freeze_test_sha
from src.services.curation.export_support import frozen_test_sha_of


if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from fastapi.testclient import TestClient


# =============================================================================
# POST /bakeoff/run
# =============================================================================


def test_dataset_id_rejects_traversal(env: dict[str, Path], client: TestClient) -> None:
    make_export(env['exports'].parent, 'outside', nc=1, test_labels={'a': [0]})
    make_run(env, 'r1', export_dir=env['exports'].parent / 'outside')
    for bad in ('export:../outside', 'export:..', 'external:../../x', 'export:/etc', 'nope:x'):
        r = _post_run(
            client, {'datasets': [{'id': bad}], 'models': [{'source': 'run', 'run_id': 'r1'}]}
        )
        assert r.status_code == 400, (bad, r.text)
    assert not env['jobs'].exists()


def test_run_resolves_run_model_class_map(env: dict[str, Path], client: TestClient) -> None:
    from src.routers.curation._bakeoff_models import BakeoffJobSpec

    d = _live_export(env)
    ckpt = make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP, imgsz=640)
    r = _post_run(
        client,
        {
            'job_id': 'j1',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'run5'}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body['status'], body['job_id'], body['profile']) == ('enqueued', 'j1', 'generic')
    assert body['datasets'] == [
        {
            'id': 'export:20260924T233203Z',
            'path': str(d),
            'frozen_test_sha': frozen_test_sha_of(d),
            'test_label_sha': freeze_test_sha(d)[0],
            'n_eval_classes': 5,
        }
    ]
    [m] = body['models']
    assert (m['model'], m['display_name'], m['source']) == ('run:run5', 'run5', 'run')
    cm = m['class_mapping']['export:20260924T233203Z']
    assert cm['method'] == 'run_class_remap'
    assert cm['model_to_eval'] == {'0': 37, '1': 38, '2': 43, '3': 51, '4': 78}
    assert cm['unmapped_model_classes'] == []
    assert cm['not_covered_eval_classes'] == []
    assert m['train_test_overlap'] == {'export:20260924T233203Z': {'n_images': 0, 'fraction': 0.0}}

    spec = BakeoffJobSpec.model_validate(json.loads((env['jobs'] / 'j1.job.json').read_text()))
    assert spec.schema_version == 2
    assert spec.out_dir == str(env['out'] / 'j1')
    [ds] = spec.datasets
    assert (ds.id, ds.dir_name, ds.path) == (
        'export:20260924T233203Z',
        'export__20260924T233203Z',
        str(d),
    )
    assert ds.eval_class_ids == [37, 38, 43, 51, 78]
    assert ds.test_label_sha == freeze_test_sha(d)[0]
    [jm] = spec.models
    assert (jm.model, jm.backend, jm.weights, jm.imgsz, jm.mode) == (
        'run:run5',
        'ultralytics',
        str(ckpt),
        640,
        'full',
    )
    assert jm.class_map_by_dataset['export:20260924T233203Z']['model_to_eval'] == {
        '0': 37,
        '1': 38,
        '2': 43,
        '3': 51,
        '4': 78,
    }


def test_run_subset_model_reports_not_covered_at_enqueue(
    env: dict[str, Path], client: TestClient
) -> None:
    d = _live_export(env)
    remap3 = {
        'original_to_new': {'38': 0, '39': 1, '44': 2},
        'new_to_original': {'0': 38, '1': 39, '2': 44},
        'single_cls': False,
        'names': ['miata', 'minicooper', 'mustang'],
        'include_classes': [38, 39, 44],
    }
    make_run(env, 'run3', export_dir=d, class_remap=remap3, include_classes=[38, 39, 44])
    r = _post_run(
        client,
        {
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'run3', 'display_name': '3-class subset'}],
        },
    )
    assert r.status_code == 200, r.text
    [m] = r.json()['models']
    assert m['display_name'] == '3-class subset'
    cm = m['class_mapping']['export:20260924T233203Z']
    assert cm['model_to_eval'] == {'0': 37, '1': 38, '2': 43}
    assert cm['not_covered_eval_classes'] == [
        {'eval_class_id': 51, 'name': 'porsche'},
        {'eval_class_id': 78, 'name': 'vw'},
    ]


def test_run_full_export_model_maps_by_registry_ids(
    env: dict[str, Path], client: TestClient
) -> None:
    d = _live_export(env)
    make_run(env, 'full', export_dir=d)
    r = _post_run(
        client,
        {
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'full'}],
        },
    )
    assert r.status_code == 200, r.text
    cm = r.json()['models'][0]['class_mapping']['export:20260924T233203Z']
    assert cm['method'] == 'registry_ids'
    assert {k: cm['model_to_eval'][k] for k in ('37', '51')} == {'37': 37, '51': 51}


def test_run_rejects_multiclass_single_cls_run(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env)
    remap = {
        'original_to_new': {'38': 0, '39': 0},
        'new_to_original': {},
        'single_cls': True,
        'names': ['object'],
        'include_classes': [38, 39],
    }
    make_run(env, 'sc', export_dir=d, class_remap=remap, single_cls=True, include_classes=[38, 39])
    r = _post_run(
        client,
        {
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'sc'}],
        },
    )
    assert r.status_code == 422
    assert r.json()['detail'] == (
        'single_cls run over 2 classes cannot be scored per class; '
        'compare it on a single-class export'
    )
    assert not env['jobs'].exists()


def test_run_alias_run_dataset_id(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env)
    make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP)
    r = _post_run(
        client,
        {'datasets': [{'id': 'run:run5'}], 'models': [{'source': 'run', 'run_id': 'run5'}]},
    )
    assert r.status_code == 200, r.text
    assert [ds['id'] for ds in r.json()['datasets']] == ['export:20260924T233203Z']

    make_run(env, 'elsewhere', export_dir=env['exports'].parent / 'not_under_root')
    r = _post_run(
        client,
        {'datasets': [{'id': 'run:elsewhere'}], 'models': [{'source': 'run', 'run_id': 'run5'}]},
    )
    assert r.status_code == 400
    assert 'export root' in r.json()['detail']


def test_run_unknown_run_or_baseline_400(env: dict[str, Path], client: TestClient) -> None:
    _live_export(env)
    ds = [{'id': 'export:20260924T233203Z'}]
    for model, needle in (
        ({'source': 'run', 'run_id': 'no_such_run'}, 'no_such_run'),
        ({'source': 'baseline', 'name': 'no_such_baseline'}, 'no_such_baseline'),
    ):
        r = _post_run(client, {'datasets': ds, 'models': [model]})
        assert r.status_code == 400, r.text
        assert needle in r.json()['detail']
    r = _post_run(
        client,
        {
            'datasets': [{'id': 'export:missing'}],
            'models': [{'source': 'custom', 'name': 'c', 'backend': 'ultralytics'}],
        },
    )
    assert r.status_code == 400
    assert 'export:missing' in r.json()['detail']
    assert not env['jobs'].exists()


def test_run_requires_models_and_datasets(env: dict[str, Path], client: TestClient) -> None:
    _live_export(env)
    r = _post_run(client, {'datasets': [{'id': 'export:20260924T233203Z'}]})
    assert r.status_code == 400
    r = _post_run(client, {'models': [{'source': 'custom', 'name': 'c', 'backend': 'ultralytics'}]})
    assert r.status_code == 400


def test_run_rejects_removed_v1_fields(env: dict[str, Path], client: TestClient) -> None:
    for body in (
        {'dataset': '/d', 'models': [{'source': 'custom', 'name': 'c', 'backend': 'ultralytics'}]},
        {'datasets': [{'path': '/d'}], 'models': []},
        {'datasets': [{'id': 'export:x'}], 'models': [{'backend': 'ultralytics', 'name': 'm'}]},
        {'datasets': [{'id': 'export:x'}], 'quantize': {'run_id': 'r', 'coreml': True}},
    ):
        assert _post_run(client, body).status_code == 422, body


def test_run_duplicate_model_keys_400(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env)
    make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP)
    r = _post_run(
        client,
        {
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [
                {'source': 'run', 'run_id': 'run5'},
                {'source': 'run', 'run_id': 'run5', 'display_name': 'again'},
            ],
        },
    )
    assert r.status_code == 400
    assert 'run:run5' in r.json()['detail']


def test_run_custom_model_explicit_and_name_mapping(
    env: dict[str, Path], client: TestClient
) -> None:
    _live_export(env)
    r = _post_run(
        client,
        {
            'job_id': 'jc',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [
                {
                    'source': 'custom',
                    'name': 'served',
                    'backend': 'triton',
                    'triton_model': 'my_detector',
                    'class_map': {'0': 'miata', '1': 'VW'},
                },
                {'source': 'custom', 'name': 'coco', 'backend': 'ultralytics', 'weights': 'w.pt'},
            ],
        },
    )
    assert r.status_code == 200, r.text
    by_key = {m['model']: m for m in r.json()['models']}
    explicit = by_key['custom:served']['class_mapping']['export:20260924T233203Z']
    assert (explicit['method'], explicit['model_to_eval']) == ('explicit', {'0': 37, '1': 78})
    names = by_key['custom:coco']['class_mapping']['export:20260924T233203Z']
    assert (names['method'], names['model_to_eval']) == ('names', None)
    assert names['warnings'] == ['resolved by name inside the evaluator']
    assert by_key['custom:coco']['train_test_overlap'] == {'export:20260924T233203Z': None}
    spec = json.loads((env['jobs'] / 'jc.job.json').read_text())
    coco = next(m for m in spec['models'] if m['model'] == 'custom:coco')
    assert coco['class_map_by_dataset'] == {'export:20260924T233203Z': None}
    served = next(m for m in spec['models'] if m['model'] == 'custom:served')
    assert served['triton_model'] == 'my_detector'


def test_run_quantize_block_filled_from_run(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env)
    ckpt = make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP, imgsz=512)
    r = _post_run(
        client,
        {
            'job_id': 'run5_quant',
            'datasets': [{'id': 'run:run5'}],
            'models': [{'source': 'run', 'run_id': 'run5'}],
            'quantize': {'run_id': 'run5', 'throughput': True},
        },
    )
    assert r.status_code == 200, r.text
    q = json.loads((env['jobs'] / 'run5_quant.job.json').read_text())['quantize']
    assert q['model_key_prefix'] == 'run:run5'
    assert (q['checkpoint'], q['imgsz'], q['calib_dataset']) == (str(ckpt), 512, str(d))
    assert q['formats'] == ['fp32_onnx', 'fp16_onnx', 'int8_onnx']
    assert (q['n_calib'], q['calib_split'], q['throughput']) == (1000, 'train', True)
    assert q['out_root'] == str(env['out'] / 'run5_quant' / 'quant')
    assert q['class_map_by_dataset']['export:20260924T233203Z']['model_to_eval']['4'] == 78


def test_run_writes_queued_status(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env)
    make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP)
    r = _post_run(
        client,
        {
            'job_id': 'jq',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'run5'}],
        },
    )
    assert r.status_code == 200, r.text
    st = client.get('/curation/projects/default/bakeoff/status/jq')
    assert st.status_code == 200
    body = st.json()
    assert body['schema_version'] == 2
    assert body['state'] == 'queued'
    assert body['datasets'] == ['export:20260924T233203Z']
    assert body['models'] == ['run:run5']
    assert body['progress'] == {'done': 0, 'total': 1}
    runs = client.get('/curation/projects/default/bakeoff/runs').json()['runs']
    assert [(x['job_id'], x['state']) for x in runs] == [('jq', 'queued')]
    # Re-using a job id would overwrite that job's results.
    again = _post_run(
        client,
        {
            'job_id': 'jq',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'run5'}],
        },
    )
    assert again.status_code == 409


def test_run_stop_failure_returns_409_and_removes_job(
    env: dict[str, Path], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.services.training.gpu_arbiter as arbiter

    async def _fail(**_kw: Any) -> None:
        raise arbiter.GpuArbiterStopFailedError('docker socket unavailable')

    monkeypatch.setattr(arbiter, 'stop_gpu_services', _fail)
    d = _live_export(env)
    make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP)
    r = _post_run(
        client,
        {
            'job_id': 'jf',
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'run5'}],
        },
    )
    assert r.status_code == 409
    assert 'docker socket unavailable' in r.json()['detail']
    assert not (env['jobs'] / 'jf.job.json').exists()
    assert list(env['jobs'].glob('*.job.json')) == []
    st = client.get('/curation/projects/default/bakeoff/status/jf').json()
    assert st['state'] == 'error'
    assert 'docker socket unavailable' in st['error']


def test_run_warns_about_train_test_overlap(env: dict[str, Path], client: TestClient) -> None:
    d = _live_export(env, train_labels={'img_37_0': [37], 'img_38_1': [38]})
    make_run(env, 'run5', export_dir=d, class_remap=LIVE_REMAP)
    r = _post_run(
        client,
        {
            'datasets': [{'id': 'export:20260924T233203Z'}],
            'models': [{'source': 'run', 'run_id': 'run5'}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['models'][0]['train_test_overlap'] == {
        'export:20260924T233203Z': {'n_images': 2, 'fraction': 2 / 25}
    }
    assert body['warnings'] == [
        (
            'run:run5: 2 of the test images of export:20260924T233203Z are in its '
            'training/validation splits'
        )
    ]
