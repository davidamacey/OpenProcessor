"""``POST /train/promote/{job_id}`` as a background job (#87).

The Triton handoff itself is faked (``promote_yolo26_to_triton``); these tests
cover the contract around it: 202 + promote_id, phases, failure, the
single-writer state under the train jobs dir, double-promote idempotency,
``wait=true`` staying synchronous, and restart repair.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest
from _project_paths import default_train_jobs_dir
from fastapi import FastAPI
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from pathlib import Path

RUN = 'pj-run'
BASE = '/curation/projects/default/train'


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from unittest.mock import AsyncMock

    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path))
    from _curation_app import mount_curation_routers

    from src.routers.curation._common import _raw_opensearch_dep
    from src.routers.curation_train import router

    app = FastAPI()
    mount_curation_routers(app, router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: AsyncMock()

    jobs_dir = default_train_jobs_dir(tmp_path)
    snapshot = jobs_dir / f'{RUN}.registry_snapshot.json'
    snapshot.write_text(
        json.dumps(
            {
                'version': 1,
                'updated_at': '2026-09-01T00:00:00Z',
                'classes': [{'class_id': 0, 'class_name': 'car', 'deprecated': False}],
            }
        )
    )
    (jobs_dir / f'{RUN}.job.json').write_text(
        json.dumps({'job_id': RUN, 'registry_snapshot_path': str(snapshot)})
    )

    from src.services.training.job_models import TrainJobStatus

    status = TrainJobStatus(
        job_id=RUN,
        state='finished',
        checkpoint_path=f'/jobs/{RUN}/best.pt',
        eval={'map50': 0.9, 'per_class': []},
    )

    async def _read_status(jid: str) -> TrainJobStatus | None:
        return status if jid == RUN else None

    monkeypatch.setattr('src.services.training.jobs.read_status', _read_status)
    with TestClient(app) as c:
        yield c


class _FakePromote:
    """Walks the real phases, parking at ``building`` until released."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, raises: Exception | None = None):
        self.release = threading.Event()
        self.calls = 0
        self.raises = raises
        monkeypatch.setattr(
            'src.services.training.triton_promote.promote_yolo26_to_triton', self, raising=True
        )

    async def __call__(self, **kwargs: Any) -> Any:
        import asyncio

        from src.services.training.triton_promote import PromoteResult

        self.calls += 1
        on_phase = kwargs.get('on_phase') or (lambda _p: None)
        on_phase('exporting')
        on_phase('loading')
        on_phase('building')
        await asyncio.to_thread(self.release.wait, 15)
        if self.raises is not None:
            raise self.raises
        on_phase('warming')
        name = kwargs['triton_name']
        return PromoteResult(
            job_id=RUN,
            triton_name=name,
            onnx_path=f'/m/{name}/1/model.onnx',
            config_path=f'/m/{name}/config.pbtxt',
            labels_path=f'/m/{name}/labels.txt',
            triton_loaded=True,
            cold_start_expected_on_first_inference=False,
        )


def _poll(client: TestClient, promote_id: str, want: set[str], timeout: float = 10) -> dict:
    deadline = time.time() + timeout
    body: dict = {}
    while time.time() < deadline:
        r = client.get(f'{BASE}/promote/{RUN}/jobs/{promote_id}')
        assert r.status_code == 200, r.text
        body = r.json()
        if body['status'] in want:
            return body
        time.sleep(0.05)
    raise AssertionError(f'never reached {want}; last={body}')


def _post(client: TestClient, name: str = 'pj_model', **params: Any):
    return client.post(f'{BASE}/promote/{RUN}', json={'triton_name': name}, params=params or None)


def test_default_returns_202_and_walks_the_phases_to_done(client, monkeypatch) -> None:
    fake = _FakePromote(monkeypatch)
    r = _post(client)
    assert r.status_code == 202, r.text
    first = r.json()
    assert first['job_id'] == RUN
    assert first['triton_name'] == 'pj_model'
    assert first['promote_id'].startswith('pm_')
    assert first['result'] is None

    parked = _poll(client, first['promote_id'], {'building'})
    assert parked['poll_after_s'] is not None
    assert parked['finished_at'] is None

    fake.release.set()
    done = _poll(client, first['promote_id'], {'done', 'failed'})
    assert done['status'] == 'done', done
    assert done['error'] is None
    assert done['result']['triton_name'] == 'pj_model'
    assert done['result']['triton_loaded'] is True
    assert done['poll_after_s'] is None
    assert done['finished_at']


def test_failure_is_a_terminal_state_with_the_sync_status_code(client, monkeypatch) -> None:
    from src.services.training.triton_promote import ModelNameConflictError

    fake = _FakePromote(monkeypatch, raises=ModelNameConflictError('pj_model'))
    fake.release.set()
    pid = _post(client).json()['promote_id']
    body = _poll(client, pid, {'failed', 'done'})
    assert body['status'] == 'failed'
    assert body['error_status'] == 409
    assert 'pj_model' in body['error']
    assert body['result'] is None


def test_second_promote_of_the_same_run_returns_the_active_job(client, monkeypatch) -> None:
    fake = _FakePromote(monkeypatch)
    first = _post(client).json()
    _poll(client, first['promote_id'], {'building'})

    again = _post(client)
    assert again.status_code == 200
    assert again.json()['promote_id'] == first['promote_id']
    assert fake.calls == 1

    other = _post(client, name='pj_other')
    assert other.status_code == 409
    detail = other.json()['detail']
    assert detail['code'] == 'promote_in_progress'
    assert detail['promote_id'] == first['promote_id']

    fake.release.set()
    _poll(client, first['promote_id'], {'done'})
    # terminal: a new promote is a new job
    fake2 = _FakePromote(monkeypatch)
    fake2.release.set()
    third = _post(client)
    assert third.status_code == 202
    assert third.json()['promote_id'] != first['promote_id']


def test_wait_true_is_synchronous_and_keeps_the_old_shape(client, monkeypatch) -> None:
    fake = _FakePromote(monkeypatch)
    fake.release.set()
    r = _post(client, wait='true')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['triton_name'] == 'pj_model'
    assert body['triton_loaded'] is True
    assert 'promote_id' not in body


def test_validation_errors_stay_synchronous(client, monkeypatch) -> None:
    _FakePromote(monkeypatch)
    r = client.post(f'{BASE}/promote/nope', json={'triton_name': 'x'})
    assert r.status_code == 404


def test_unknown_or_foreign_promote_id_is_404(client, monkeypatch) -> None:
    fake = _FakePromote(monkeypatch)
    fake.release.set()
    pid = _post(client).json()['promote_id']
    assert client.get(f'{BASE}/promote/{RUN}/jobs/pm_nope').status_code == 404
    assert client.get(f'{BASE}/promote/other-run/jobs/{pid}').status_code == 404
    assert client.get(f'{BASE}/promote/{RUN}/jobs/..%2Fx').status_code in (404, 422)


def test_train_status_surfaces_the_latest_promote(client, monkeypatch) -> None:
    fake = _FakePromote(monkeypatch)
    assert client.get(f'{BASE}/status/{RUN}').json()['promote'] is None
    pid = _post(client).json()['promote_id']
    _poll(client, pid, {'building'})
    promote = client.get(f'{BASE}/status/{RUN}').json()['promote']
    assert promote['promote_id'] == pid
    assert promote['status'] == 'building'
    fake.release.set()


def test_a_job_left_active_by_a_dead_process_reads_failed(client, tmp_path) -> None:
    from src.services.training import promote_job

    job, created = promote_job.claim(run_job_id=RUN, triton_name='pj_model')
    assert created
    old = time.time() - 600
    os.utime(job.heartbeat_file, (old, old))  # process died; heartbeat stale

    body = client.get(f'{BASE}/promote/{RUN}/jobs/{job.directory.name}').json()
    assert body['status'] == 'failed'
    assert 'heartbeat stale' in body['error']
    # and a fresh promote is no longer blocked by it
    again, created = promote_job.claim(run_job_id=RUN, triton_name='pj_model')
    assert created
    assert again.directory != job.directory


@pytest.mark.asyncio
async def test_aborted_wait_does_not_cancel_the_promote() -> None:
    """A proxy timeout cancels the awaiting request; the load/warm-up it started must
    still finish, or Triton is left with a half-loaded model."""
    import asyncio

    from src.services.training import promote_job

    finished: list[str] = []
    gate = asyncio.Event()

    async def _promote() -> str:
        await gate.wait()
        finished.append('warmed')
        return 'ok'

    request = asyncio.create_task(promote_job.run_detached(_promote()))
    await asyncio.sleep(0)
    request.cancel()  # the 504: the handler awaiting the promote is cancelled
    with pytest.raises(asyncio.CancelledError):
        await request
    assert finished == []
    gate.set()
    await asyncio.sleep(0.05)
    assert finished == ['warmed']


@pytest.mark.asyncio
async def test_detached_promote_still_returns_and_raises_to_a_live_caller() -> None:
    from src.services.training import promote_job

    async def _ok() -> int:
        return 7

    async def _boom() -> int:
        msg = 'load refused'
        raise RuntimeError(msg)

    assert await promote_job.run_detached(_ok()) == 7
    with pytest.raises(RuntimeError, match='load refused'):
        await promote_job.run_detached(_boom())
