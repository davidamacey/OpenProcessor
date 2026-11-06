"""The ``train`` handler with a fake fit: progress per epoch, cancel at a boundary, failure text.

The fit (Ultralytics) is replaced by a function that moves the same ``StatusState`` the epoch
callbacks move; everything around it (spec parsing, progress, the run log, the manifest, the
artifacts, the campaign hook) is the real handler.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import threading
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import celery_tasks
import job_spec
import pytest

from src.clients.storage.roots import RootedStore
from src.db.repos.training_runs import RunRow


EPOCHS = 3


class FakeSession:
    async def run_sync(self, fn: Any) -> Any:
        return fn(None)


class FakeCtx:
    def __init__(self, job_id: uuid.UUID) -> None:
        self.job_id = job_id
        self.progress_calls: list[tuple[int, int | None, str | None]] = []
        self.cancel = threading.Event()

    def cancelled(self) -> bool:
        return self.cancel.is_set()

    async def progress(self, done, total=None, stage=None, counters=None, *, force=False) -> None:
        del counters, force
        self.progress_calls.append((done, total, stage))

    @asynccontextmanager
    async def session(self):
        yield FakeSession()


@pytest.fixture
def world(tmp_path, monkeypatch):
    export = tmp_path / 'exports' / 'e1'
    export.mkdir(parents=True)
    store = RootedStore(
        {n: tmp_path / n for n in ('projects', 'source', 'models', 'cache', 'upgrade')}
    )
    monkeypatch.setattr(celery_tasks, 'build_rooted_store', lambda _paths: store)
    monkeypatch.setattr(celery_tasks, 'storage_root_paths', lambda: None)
    monkeypatch.setattr(celery_tasks, 'POLL_INTERVAL_S', 0.02)
    monkeypatch.setattr(job_spec, 'TMP_ROOT', tmp_path / 'scratch')

    job_id = uuid.uuid4()
    run = RunRow(
        job_id=job_id,
        project_id=uuid.uuid4(),
        spec={
            'model_family': 'yolo26',
            'model_size': 'n',
            'profile': 'probe',
            'dataset_export_dir': str(export),
            'project_export_root': str(tmp_path / 'exports'),
            'hyperparameters': {'epochs': EPOCHS},
        },
        class_names=['a', 'b'],
        family='yolo26',
        base_model='yolo26n',
        artifact_prefix=f'projects/p1/training/{job_id}',
        manifest=None,
        mlflow_run_id=None,
        gpu_uuids=[],
        status='running',
        params={},
        progress={},
        error_code=None,
        error_message=None,
        created_at=dt.datetime.now(dt.UTC),
        started_at=None,
        finished_at=None,
        heartbeat_at=None,
    )
    stored: dict[str, Any] = {}

    class FakeRepo:
        @staticmethod
        def get(_s, _job_id):
            return run

        @staticmethod
        def set_manifest(_s, _job_id, manifest, mlflow_run_id):
            stored.update(manifest=manifest, mlflow_run_id=mlflow_run_id)

    held: dict[str, Any] = {'reserved': [], 'released': 0, 'while_fitting': None}

    class FakeReservations:
        @staticmethod
        def reserve(_s, _job_id, cards):
            held['reserved'] = [c.gpu_uuid for c in cards]

        @staticmethod
        def release(_s, _job_id):
            held['released'] += 1

    class FakeDevices:
        @staticmethod
        def on_host_at(_s, _host, positions):
            held['positions'] = list(positions)
            return [SimpleNamespace(gpu_uuid=f'GPU-{p}') for p in positions]

    FakeRepo.set_gpu_uuids = staticmethod(lambda _s, _job_id, uuids: stored.update(gpu=uuids))  # type: ignore[attr-defined]
    monkeypatch.setattr(celery_tasks, 'runs_repo', FakeRepo)
    monkeypatch.setattr(celery_tasks, 'reservations', FakeReservations)
    monkeypatch.setattr(celery_tasks, 'gpu_devices', FakeDevices)
    stored['held'] = held
    monkeypatch.setattr(celery_tasks, 'maybe_handle_campaign', _no_campaign)
    ctx = FakeCtx(job_id)
    return ctx, store, run, stored


async def _no_campaign(*_args: Any) -> None:
    return None


def _fake_fit(*, fail_at: int | None = None, epoch_s: float = 0.05):
    def fit(spec, state) -> None:
        print('[fit] start')
        with state.lock:
            state.state = 'running'
            state.total_epochs = EPOCHS
        import time

        for epoch in range(1, EPOCHS + 1):
            time.sleep(epoch_s)
            if fail_at == epoch:
                raise RuntimeError('CUDA exploded')
            with state.lock:
                state.current_epoch = epoch
                state.last_epoch_metric = {'epoch': epoch, 'map50': 0.1 * epoch}
            if state.cancel_requested.is_set():
                with state.lock:
                    state.state = 'cancelled'
                return
        with state.lock:
            state.state = 'finished'
            state.eval = {'map50': 0.3}

    return fit


def test_a_fake_fit_completes_with_progress_per_epoch(world, monkeypatch) -> None:
    ctx, store, run, stored = world
    monkeypatch.setattr(celery_tasks, 'run_training', _fake_fit())

    result = asyncio.run(celery_tasks.train(ctx))

    assert result['state'] == 'finished'
    epochs = [done for done, _total, _stage in ctx.progress_calls]
    assert max(epochs) == EPOCHS
    assert epochs == sorted(epochs)  # never goes backwards
    assert {1, 2}.issubset(set(epochs))  # reported while running, not only at the end
    assert stored['manifest']['results']['eval'] == {'map50': 0.3}
    log = store.open_read(f'{run.artifact_prefix}/run.log').read().decode()
    assert '[fit] start' in log


def test_a_cancel_stops_at_an_epoch_boundary_and_the_handler_returns(world, monkeypatch) -> None:
    ctx, _store, _run, stored = world
    monkeypatch.setattr(celery_tasks, 'run_training', _fake_fit(epoch_s=0.2))

    async def go() -> dict:
        task = asyncio.ensure_future(celery_tasks.train(ctx))
        await asyncio.sleep(0.3)
        ctx.cancel.set()
        return await task

    result = asyncio.run(go())

    assert result['state'] == 'cancelled'
    assert max(done for done, _t, _s in ctx.progress_calls) < EPOCHS
    assert stored['manifest']['results']['final_state'] == 'cancelled'


def test_an_exception_fails_the_job_with_its_message_and_keeps_the_log(world, monkeypatch) -> None:
    ctx, store, run, stored = world
    monkeypatch.setattr(celery_tasks, 'run_training', _fake_fit(fail_at=2))

    with pytest.raises(RuntimeError, match='CUDA exploded'):
        asyncio.run(celery_tasks.train(ctx))

    log = store.open_read(f'{run.artifact_prefix}/run.log').read().decode()
    assert 'FAILED: CUDA exploded' in log
    assert stored['manifest']['results']['final_state'] == 'failed'


def test_the_cards_are_held_by_uuid_during_the_fit_and_released_after_it(
    world, monkeypatch
) -> None:
    ctx, _store, run, stored = world
    run.spec['cuda_visible_devices'] = '2'
    monkeypatch.setenv('OP_TRAIN_GPU_ORDER', '0,2')  # host card 2 is position 1 in this container
    seen: dict[str, Any] = {}
    base_fit = _fake_fit()

    def fit(spec, state) -> None:
        seen['reserved_during_fit'] = list(stored['held']['reserved'])
        seen['released_during_fit'] = stored['held']['released']
        base_fit(spec, state)

    monkeypatch.setattr(celery_tasks, 'run_training', fit)

    asyncio.run(celery_tasks.train(ctx))

    assert stored['held']['positions'] == [1]
    assert seen == {'reserved_during_fit': ['GPU-1'], 'released_during_fit': 0}
    assert stored['gpu'] == ['GPU-1']
    assert stored['held']['released'] == 1


def test_the_cards_are_released_when_the_fit_fails(world, monkeypatch) -> None:
    ctx, _store, run, stored = world
    run.spec['cuda_visible_devices'] = '0'
    monkeypatch.setattr(celery_tasks, 'run_training', _fake_fit(fail_at=1))

    with pytest.raises(RuntimeError, match='CUDA exploded'):
        asyncio.run(celery_tasks.train(ctx))

    assert stored['held']['released'] == 1


def test_a_card_the_container_is_not_attached_to_fails_the_run_before_the_fit(
    world, monkeypatch
) -> None:
    ctx, _store, run, stored = world
    run.spec['cuda_visible_devices'] = '1'
    monkeypatch.setenv('OP_TRAIN_GPU_ORDER', '0,2')
    fitted = []
    monkeypatch.setattr(celery_tasks, 'run_training', lambda *_a: fitted.append(True))

    with pytest.raises(ValueError, match='not attached'):
        asyncio.run(celery_tasks.train(ctx))

    assert fitted == []
    assert stored['manifest']['results']['final_state'] == 'failed'
