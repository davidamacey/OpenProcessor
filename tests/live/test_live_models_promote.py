"""Live: model upload, re-export and promote, to the depth the harness allows.

What is real: the API's checkpoint validation (Ultralytics' restricted loader, on a tiny untrained
detector the test builds inside the API image), the ``model_export`` job and its status, the
``models`` registry version, and the promote route's gates against a run the fake trainer
finished. What is not: the TensorRT build. ``export-worker`` is ``fake_consumers.py``'s
``model_export`` handler, which completes a job without building an engine, so ``triton_loaded``
stays false and no engine exists; likewise no ONNX is written for a promote, so a forced promote
stops at the ONNX check. Both end states are asserted, not skipped.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from .conftest import API_PREFIX, VERIFY_DATA_DIR, compose, wait_for_state


pytestmark = pytest.mark.live

TERMINAL_EXPORT = ('completed', 'failed', 'cancelled')
# The API's own YOLO checkpoint, with no download: build from the architecture file and save.
_BUILD_TINY_CHECKPOINT = 'from ultralytics import YOLO; m = YOLO("yolo11n.yaml"); m.save("/verify-data/images/{name}.pt")'


@pytest.fixture(scope='module')
def tiny_checkpoint() -> bytes:
    name = f'tiny{uuid.uuid4().hex[:6]}'
    proc = compose(
        'exec',
        '-T',
        'api',
        'python3',
        '-c',
        _BUILD_TINY_CHECKPOINT.format(name=name),
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    path = VERIFY_DATA_DIR / 'images' / f'{name}.pt'
    try:
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def _export(api: Any, task_id: str) -> dict[str, Any]:
    def fetch() -> dict[str, Any]:
        resp = api.get(f'/models/export/{task_id}')
        assert resp.status_code == 200, resp.text
        return resp.json()

    return wait_for_state(fetch, lambda t: t['status'] in TERMINAL_EXPORT, timeout=120)


def test_upload_refuses_a_file_that_is_not_a_checkpoint(api: Any) -> None:
    exports_before = api.get('/models/exports').json()
    not_pt = api.post('/models/upload', files={'file': ('weights.bin', b'x' * 4000)})
    assert not_pt.status_code == 400, not_pt.text
    junk = api.post(
        '/models/upload', files={'file': ('junk.pt', b'this is not a checkpoint' * 100)}
    )
    assert junk.status_code == 400, junk.text
    assert api.get('/models/exports').json() == exports_before, 'a refused upload queued a job'


def test_upload_validates_queues_and_completes_through_the_worker(
    api: Any, tiny_checkpoint: bytes
) -> None:
    name = f'live_up_{uuid.uuid4().hex[:6]}'
    resp = api.post(
        '/models/upload',
        files={'file': (f'{name}.pt', tiny_checkpoint)},
        data={'triton_name': name, 'formats': 'onnx', 'auto_load': 'false'},
    )
    assert resp.status_code == 200, resp.text
    started = resp.json()
    assert started['triton_name'] == name
    assert started['status'] == 'pending'

    task = _export(api, started['task_id'])
    assert task['status'] == 'completed', task
    assert task['triton_name'] == name
    # The validator read the checkpoint: the class names of the architecture's default dataset.
    assert task['num_classes'] == 80, task
    assert task['triton_loaded'] is False
    assert started['task_id'] in {t['task_id'] for t in api.get('/models/exports').json()}

    # A re-export of the stored weights is another job on the same worker.
    again = api.post(f'/models/{name}/export', json={'formats': ['onnx'], 'auto_load': False})
    assert again.status_code == 200, again.text
    assert again.json()['task_id'] != started['task_id']
    assert _export(api, again.json()['task_id'])['status'] == 'completed'


def test_export_refuses_a_bad_name_and_an_unknown_model(api: Any) -> None:
    assert api.post('/models/Bad Name/export', json={}).status_code in (400, 404, 422)
    unknown = api.post(f'/models/never_{uuid.uuid4().hex[:6]}/export', json={})
    assert unknown.status_code == 404, unknown.text


def _finished_run(api: Any) -> str:
    exported = api.post(f'{API_PREFIX}/projects/default/export/yolo', json={'version_tag': 'promo'})
    assert exported.status_code == 200, exported.text
    started = api.post(
        f'{API_PREFIX}/projects/default/train/start',
        params={'force': True},
        json={
            'dataset_export_dir': '/verify-data/projects/default/exports/current',
            'model_size': 'n',
            'profile': 'probe',
            'cuda_visible_devices': '0',
        },
    )
    assert started.status_code == 201, started.text
    job_id = started.json()['job_id']
    state = wait_for_state(
        lambda: api.get(f'{API_PREFIX}/projects/default/train/status/{job_id}').json(),
        lambda s: bool(s) and s.get('state') == 'finished',
        timeout=120,
    )
    assert state['state'] == 'finished', state
    return str(job_id)


def test_promote_is_gated_then_stops_at_the_missing_onnx(api: Any) -> None:
    job_id = _finished_run(api)
    base = f'{API_PREFIX}/projects/default/train'
    body = {'triton_name': f'promoted_{uuid.uuid4().hex[:6]}'}

    gated = api.post(f'{base}/promote/{job_id}', json=body)
    assert gated.status_code == 422, gated.text
    detail = gated.json()['detail']
    assert detail['failures'], detail
    assert detail['force_allowed'] is True, detail

    forced = api.post(f'{base}/promote/{job_id}', params={'force': True}, json=body)
    # The fake trainer wrote no ONNX export: forced past the score gate, the route still refuses
    # (404) or the job ends failed, and nothing is left promoted.
    if forced.status_code == 202:
        promote_id = forced.json()['promote_id']
        final = wait_for_state(
            lambda: api.get(f'{base}/promote/{job_id}/jobs/{promote_id}').json(),
            lambda s: s.get('status') in ('done', 'failed'),
            timeout=120,
        )
        assert final['status'] == 'failed', final
        assert final['error'], final
    else:
        assert forced.status_code in (404, 422), forced.text
    manifest = api.get(f'{base}/manifest/{job_id}').json()
    assert manifest.get('promoted_to') is None, manifest
