"""The job.json / status.json / run-log / pending-queue file contract between the API half and
the trainer half of the protocol (API writer -> trainer parser, trainer writer -> API reader).
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import job_protocol
import job_spec
import job_status
import pytest
import trainer

from curation.trainer_protocol._helpers import _write_job
from src.services.training import job_files, jobs as train_jobs


if TYPE_CHECKING:
    from pathlib import Path


def test_api_written_job_json_parses_in_trainer(jobs_dir: Path, export_dir: Path) -> None:
    """The exact bytes ``POST /train/start`` writes must parse in the trainer."""
    job_id = _write_job(
        dataset_export_dir=str(export_dir),
        model_size='n',
        profile='probe',
        include_classes=[5, 12],
        single_cls=False,
        cuda_visible_devices='0',
        hyperparameters={'epochs': 3, 'batch': 4, 'optimizer': 'MuSGD'},
    )

    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    assert spec.job_id == job_id
    assert spec.dataset_export_dir == export_dir
    assert spec.include_classes == [5, 12]
    assert spec.model_family == 'yolo26'
    assert spec.model_size == 'n'
    assert spec.profile == 'probe'
    assert spec.hyperparameters['epochs'] == 3
    # Lineage the API auto-fills at submit time and the manifest re-publishes.
    assert spec.raw['frozen_test_sha'] == 'sha-frozen-test'


def test_trainer_sibling_paths_match_the_api_side(jobs_dir: Path, export_dir: Path) -> None:
    """status / cancel / log / manifest paths are a shared naming contract."""
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    assert spec.status_path == job_files._status_path(job_id)
    assert spec.cancel_path == job_files._cancel_path(job_id)
    assert spec.run_log_path == job_files._log_path(job_id)
    assert spec.manifest_path == job_files._manifest_path(job_id)


def test_concurrent_status_writes_do_not_collide_on_the_tmp_file(tmp_path: Path) -> None:
    """The heartbeat thread and the main thread both write status.json; a shared
    ``.tmp`` name let one rename the other's file away, so the terminal write
    raised FileNotFoundError (a flaky cancel test, and a lost terminal status)."""
    import threading

    path = tmp_path / 'job.status.json'
    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def _writer() -> None:
        start.wait()
        for i in range(200):
            try:
                job_status._atomic_write_json(path, {'n': i})
            except OSError as exc:
                errors.append(exc)

    threads = [threading.Thread(target=_writer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert json.loads(path.read_text())['n'] == 199
    assert [p.name for p in tmp_path.iterdir()] == ['job.status.json']


def test_api_cancel_sentinel_is_seen_by_the_trainer(jobs_dir: Path, export_dir: Path) -> None:
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')
    assert not spec.cancel_path.exists()

    asyncio.run(train_jobs.write_cancel(job_id))

    assert spec.cancel_path.exists()


@pytest.mark.parametrize(
    ('mutation', 'expected'),
    [
        ({'job_id': ''}, 'job_id'),
        ({'model_family': 'yolo99'}, 'model_family'),
        ({'model_size': 'z'}, 'model_size'),
        ({'profile': 'turbo'}, 'profile'),
        ({'dataset_export_dir': ''}, 'dataset_export_dir'),
        ({'dataset_export_dir': '/nope/does/not/exist'}, 'dataset_export_dir'),
        ({'include_classes': ['five']}, 'include_classes'),
        ({'hyperparameters': {'optimizer': 'auto'}}, 'optimizer'),
    ],
)
def test_broken_job_json_is_rejected(
    jobs_dir: Path, export_dir: Path, mutation: dict[str, Any], expected: str
) -> None:
    """A hand-written or stale job file must fail loudly, not half-run."""
    payload: dict[str, Any] = {
        'job_id': 'job-broken',
        'model_family': 'yolo26',
        'model_size': 'n',
        'profile': 'probe',
        'dataset_export_dir': str(export_dir),
        'hyperparameters': {},
    }
    payload.update(mutation)
    path = jobs_dir / 'job-broken.job.json'
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=expected):
        job_spec.parse_and_validate_job(path)


def test_invalid_job_gets_a_terminal_failed_status(jobs_dir: Path) -> None:
    """Otherwise the watcher would re-read the same broken file forever."""
    (jobs_dir / 'job-bad.job.json').write_text('{not json')

    assert trainer.process_one(jobs_dir) is True

    status = asyncio.run(train_jobs.read_status('job-bad'))
    assert status is not None
    assert status.state == 'failed'
    assert 'validation' in (status.error or '')
    # Terminal -> no longer pending.
    assert job_protocol.list_pending_jobs(jobs_dir) == []


def test_trainer_status_payload_parses_in_the_api_reader(jobs_dir: Path, export_dir: Path) -> None:
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    state = job_status.StatusState(job_id=job_id, state='running')
    state.current_epoch = 2
    state.total_epochs = 3
    state.eval = {'map50': 0.71, 'map50_95': 0.42}
    state.checkpoint_path = '/runs/x/weights/best.pt'
    job_status.write_status_now(spec.status_path, state)

    status = asyncio.run(train_jobs.read_status(job_id))
    assert status is not None
    assert status.state == 'running'
    assert status.current_epoch == 2
    assert status.total_epochs == 3
    assert status.checkpoint_path == '/runs/x/weights/best.pt'
    # eval may be test-split numbers, so it is never copied into the
    # best-checkpoint (validation) field.
    assert status.best_checkpoint_metric is None
    assert status.eval is not None
    assert status.eval['map50'] == 0.71
    assert status.heartbeat_at is not None


@pytest.mark.parametrize(
    'state_name', sorted(job_status.TERMINAL_STATES | job_status.ACTIVE_STATES)
)
def test_every_trainer_state_is_accepted_by_the_api_model(
    jobs_dir: Path, export_dir: Path, state_name: str
) -> None:
    """A state the trainer can write that the API rejects would blank the UI."""
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    job_status.write_status_now(
        spec.status_path, job_status.StatusState(job_id=job_id, state=state_name)
    )

    status = asyncio.run(train_jobs.read_status(job_id))
    assert status is not None, f'API rejected trainer state {state_name!r}'
    assert status.state == state_name


def test_run_log_is_tailable_by_the_api(jobs_dir: Path, export_dir: Path) -> None:
    job_id = _write_job(dataset_export_dir=str(export_dir))
    spec = job_spec.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    with job_protocol.tee_to_run_log(spec.run_log_path):
        print('[trainer] epoch 1/3')
        print('[trainer] epoch 2/3')

    assert asyncio.run(train_jobs.tail_run_log(job_id, lines=10))[-1] == '[trainer] epoch 2/3'


def test_pending_list_skips_terminal_jobs_and_is_fifo(jobs_dir: Path, export_dir: Path) -> None:
    first = _write_job(job_id='2026-01-01T00-00-00_a', dataset_export_dir=str(export_dir))
    second = _write_job(job_id='2026-01-01T00-00-01_b', dataset_export_dir=str(export_dir))

    assert [p.name for p in job_protocol.list_pending_jobs(jobs_dir)] == [
        f'{first}.job.json',
        f'{second}.job.json',
    ]

    job_status.write_status_now(
        job_files._status_path(first), job_status.StatusState(job_id=first, state='finished')
    )
    assert [p.name for p in job_protocol.list_pending_jobs(jobs_dir)] == [f'{second}.job.json']
