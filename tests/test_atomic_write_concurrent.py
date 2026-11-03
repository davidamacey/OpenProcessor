"""Concurrent writers of one file never raise, and a status poll never writes state.

Writers are independent spawned processes (no shared lock): with a fixed temp name the second
``replace()`` finds the file gone and raises ``FileNotFoundError``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from src.services.curation import file_job as file_job_module
from src.services.curation.file_job import FileJob


PROCS = 4
WRITES = 1000


def _hammer(directory: str, n: int, queue: mp.Queue) -> None:
    job = FileJob(Path(directory))
    errors = 0
    for i in range(n):
        try:
            job.write({'status': 'running', 'i': i, 'pid': os.getpid()})
        except FileNotFoundError:
            errors += 1
    queue.put(errors)


def _run(directory: Path, procs: int) -> list[int]:
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    workers = [
        ctx.Process(target=_hammer, args=(str(directory), WRITES, queue)) for _ in range(procs)
    ]
    for w in workers:
        w.start()
    results = [queue.get(timeout=120) for _ in workers]
    for w in workers:
        w.join(timeout=30)
    return results


def test_concurrent_file_job_writers_raise_nothing(tmp_path: Path) -> None:
    assert _run(tmp_path / 'job', PROCS) == [0] * PROCS
    assert FileJob(tmp_path / 'job').read()['status'] == 'running'
    assert [p.name for p in (tmp_path / 'job').iterdir()] == ['state.json']


def test_control_single_writer_round_trips(tmp_path: Path) -> None:
    assert _run(tmp_path / 'job', 1) == [0]
    state = FileJob(tmp_path / 'job').read()
    assert state['i'] == WRITES - 1


def test_atomic_write_json_leaves_no_temp_file_on_failure(tmp_path: Path) -> None:
    from src.utils.atomic_write import atomic_write_json

    target = tmp_path / 'x.json'
    atomic_write_json(target, {'a': 1})
    with pytest.raises(TypeError):
        atomic_write_json(target, {'a': object()}, default=None)
    assert target.read_text() == '{"a": 1}'
    assert [p.name for p in tmp_path.iterdir()] == ['x.json']


@pytest.mark.parametrize('module', ['item_scores.job', 'embedding_viz', 'selection.job'])
def test_status_poll_of_stale_job_does_not_write_state(
    module: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    mod = importlib.import_module(f'src.services.curation.{module}')
    job = FileJob(tmp_path / 'job')
    monkeypatch.setattr(mod, '_job', lambda: job)
    job.write({'job_id': 'j1', 'status': 'running'})
    job.touch_heartbeat()
    old = time.time() - file_job_module.HEARTBEAT_STALE_S - 60
    os.utime(job.heartbeat_file, (old, old))
    past = time.time() - 100
    os.utime(job.state_file, (past, past))
    before = job.state_file.stat().st_mtime_ns

    state = mod.get_state()

    assert state['status'] == 'failed'  # the derived view
    assert job.state_file.stat().st_mtime_ns == before
    assert job.read()['status'] == 'running'


def test_control_fresh_heartbeat_poll_reports_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation.item_scores import job as scores

    job = FileJob(tmp_path / 'job')
    monkeypatch.setattr(scores, '_job', lambda: job)
    job.write({'job_id': 'j1', 'status': 'running'})
    job.touch_heartbeat()
    assert scores.get_state()['status'] == 'running'
