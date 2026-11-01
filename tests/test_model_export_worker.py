"""POST /models/upload and /models/{name}/export hand the engine build to the model-export
worker through job files (the API image has no TensorRT).

File protocol, under the export task dir (EXPORT_TASK_DIR, default /jobs/model_export):
  queue/<id>.job.json    written by the API (atomic rename), claimed by the worker
  claimed/<id>.job.json  the worker's claim (atomic rename out of queue/)
  <id>.json              the task status the status routes serve (written by both sides)
  worker.heartbeat       touched by the worker every couple of seconds
A missing or stale heartbeat makes the routes answer 503 naming the service.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from src.schemas.models import ExportStatus
from src.services import model_export


if TYPE_CHECKING:
    from collections.abc import Iterator

EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import export_worker  # noqa: E402


@pytest.fixture
def task_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / 'model_export'
    monkeypatch.setenv('EXPORT_TASK_DIR', str(root))
    monkeypatch.setattr(model_export, 'PYTORCH_MODELS_DIR', tmp_path / 'pytorch_models')
    model_export.export_tasks.clear()
    yield root
    model_export.export_tasks.clear()


@pytest.fixture
def client() -> TestClient:
    from src.app.factory import create_app

    return TestClient(create_app(), raise_server_exceptions=False)


def _heartbeat(root: Path, age_s: float = 0.0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    beat = root / 'worker.heartbeat'
    beat.write_text('x')
    stamp = time.time() - age_s
    os.utime(beat, (stamp, stamp))


def _pt() -> dict[str, tuple[str, bytes, str]]:
    return {'file': ('car.pt', b'x' * 2000, 'application/octet-stream')}


def _no_files(root: Path) -> bool:
    return not [p for p in root.rglob('*') if p.is_file() and p.name != 'worker.heartbeat']


@pytest.mark.parametrize('age_s', [None, 600.0])
def test_upload_without_a_live_worker_is_a_503_naming_the_service(
    client: TestClient, task_dir: Path, age_s: float | None
) -> None:
    if age_s is not None:
        _heartbeat(task_dir, age_s)
    resp = client.post('/models/upload', files=_pt())
    assert resp.status_code == 503
    assert 'model-export-worker' in resp.json()['detail']
    assert '--profile export' in resp.json()['detail']
    assert _no_files(task_dir), 'no task, no job and no saved weights without a worker'
    assert not (model_export.PYTORCH_MODELS_DIR).exists()


def test_export_route_without_a_live_worker_is_a_503_and_creates_no_task(
    client: TestClient, task_dir: Path
) -> None:
    model_export.PYTORCH_MODELS_DIR.mkdir(parents=True)
    (model_export.PYTORCH_MODELS_DIR / 'car.pt').write_bytes(b'x' * 2000)
    resp = client.post('/models/car/export', json={})
    assert resp.status_code == 503
    assert 'model-export-worker' in resp.json()['detail']
    assert _no_files(task_dir)


def test_upload_with_a_live_worker_queues_one_job_file(
    client: TestClient, task_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _heartbeat(task_dir)
    saved = model_export.PYTORCH_MODELS_DIR / 'car.pt'

    async def fake_save(content: bytes, name: str):
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(content)
        return saved, {'num_classes': 2, 'class_names': ['a', 'b'], 'end2end': False}

    monkeypatch.setattr('src.routers.models_export.save_uploaded_file', fake_save)
    resp = client.post(
        '/models/upload',
        files=_pt(),
        data={'formats': 'trt_end2end', 'max_batch': '16', 'auto_load': 'false'},
    )
    assert resp.status_code == 200, resp.text
    task_id = resp.json()['task_id']
    job = json.loads((task_dir / 'queue' / f'{task_id}.job.json').read_text())
    assert job == {
        'task_id': task_id,
        'pt_file': str(saved),
        'triton_name': 'car',
        'max_batch': 16,
        'formats': ['onnx_end2end', 'trt_end2end'],
        'normalize_boxes': True,
        'auto_load': False,
        'end2end': False,
    }
    assert model_export.get_export_task(task_id)['status'] == ExportStatus.PENDING.value
    status = client.get(f'/models/export/{task_id}')
    assert status.status_code == 200
    assert status.json()['status'] == 'pending'


def test_status_route_sees_what_the_worker_wrote_after_the_api_cached_the_task(
    task_dir: Path,
) -> None:
    task_id = model_export.create_export_task('car.pt', 'car', {}, ['trt'])
    assert model_export.get_export_task(task_id)['status'] == ExportStatus.PENDING
    on_disk = json.loads((task_dir / f'{task_id}.json').read_text())
    on_disk['status'] = ExportStatus.COMPLETED.value
    (task_dir / f'{task_id}.json').write_text(json.dumps(on_disk))
    assert model_export.get_export_task(task_id)['status'] == ExportStatus.COMPLETED.value


# --- the worker -------------------------------------------------------------------------------


def _enqueue(task_dir: Path, **over: object) -> str:
    task_id = model_export.create_export_task(
        'car.pt', 'car', {'num_classes': 2, 'end2end': bool(over.get('end2end'))}, ['trt']
    )
    model_export.enqueue_export_job(
        task_id,
        pt_file=Path('/app/pytorch_models/car.pt'),
        triton_name='car',
        max_batch=8,
        formats=['trt'],
        normalize_boxes=True,
        auto_load=bool(over.get('auto_load', False)),
    )
    return task_id


def test_claim_next_moves_the_job_out_of_the_queue_exactly_once(task_dir: Path) -> None:
    task_id = _enqueue(task_dir)
    job = export_worker.claim_next()
    assert job is not None
    assert job['task_id'] == task_id
    assert not (task_dir / 'queue' / f'{task_id}.job.json').exists()
    assert (task_dir / 'claimed' / f'{task_id}.job.json').is_file()
    assert export_worker.claim_next() is None


def test_build_command_routes_by_architecture() -> None:
    job = {
        'pt_file': '/app/pytorch_models/car.pt',
        'triton_name': 'car',
        'max_batch': 8,
        'formats': ['onnx_end2end', 'trt_end2end'],
        'normalize_boxes': True,
        'end2end': False,
    }
    legacy = export_worker.build_command(job)
    assert legacy[1].endswith('export_models.py')
    assert legacy[legacy.index('--custom-model') + 1] == '/app/pytorch_models/car.pt:car:8'
    assert '--normalize-boxes' in legacy
    assert '--generate-config' in legacy
    native = export_worker.build_command({**job, 'end2end': True})
    assert native[1].endswith('export_yolo26.py')
    assert native[native.index('--formats') + 1 :] == ['trt']


def _stub_export_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    script = tmp_path / 'stub_export.py'
    script.write_text(body)
    monkeypatch.setattr(export_worker, 'EXPORT_SCRIPT', script)


def test_run_job_completes_the_task_and_loads_the_engine(
    task_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_export_script(
        tmp_path, monkeypatch, 'print("ONNX End2End Export")\nprint("Building TensorRT engine")\n'
    )
    loaded: list[str] = []
    monkeypatch.setattr(
        export_worker, 'load_into_triton', lambda names: loaded.extend(names) or True
    )
    monkeypatch.setattr(export_worker, 'APP_DIR', tmp_path)
    task_id = _enqueue(task_dir, auto_load=True)
    job = export_worker.claim_next()
    export_worker.run_job(job)
    task = model_export.get_export_task(task_id)
    assert task['status'] == ExportStatus.COMPLETED.value
    assert task['progress'] == 100.0
    assert task['triton_loaded'] is True
    assert task['formats_completed'] == ['trt']
    assert loaded == ['car_trt']
    assert not (task_dir / 'claimed' / f'{task_id}.job.json').exists()


def test_run_job_records_the_failure_tail_and_does_not_load(
    task_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_export_script(
        tmp_path, monkeypatch, 'import sys\nprint("boom: no TensorRT")\nsys.exit(3)\n'
    )
    monkeypatch.setattr(
        export_worker,
        'load_into_triton',
        lambda _names: pytest.fail('must not load a failed export'),
    )
    monkeypatch.setattr(export_worker, 'APP_DIR', tmp_path)
    task_id = _enqueue(task_dir, auto_load=True)
    export_worker.run_job(export_worker.claim_next())
    task = model_export.get_export_task(task_id)
    assert task['status'] == ExportStatus.FAILED.value
    assert 'boom: no TensorRT' in task['error']
    assert not (task_dir / 'claimed' / f'{task_id}.job.json').exists()


def test_startup_fails_the_tasks_a_dead_worker_left_claimed(task_dir: Path) -> None:
    task_id = _enqueue(task_dir)
    assert export_worker.claim_next() is not None
    task = model_export.get_export_task(task_id)
    task['status'] = ExportStatus.EXPORTING.value
    model_export.persist_task(task_id)

    assert export_worker.reconcile_claimed() == 1
    done = model_export.get_export_task(task_id)
    assert done['status'] == ExportStatus.FAILED.value
    assert 'worker' in done['error']
    assert not (task_dir / 'claimed' / f'{task_id}.job.json').exists()
    assert export_worker.reconcile_claimed() == 0


def test_heartbeat_makes_the_api_see_a_live_worker(task_dir: Path) -> None:
    assert not model_export.export_worker_alive()
    export_worker.write_heartbeat()
    assert model_export.export_worker_alive()
    _heartbeat(task_dir, model_export.HEARTBEAT_MAX_AGE_S + 5)
    assert not model_export.export_worker_alive()
