"""Concurrent writers of one file never raise.

Writers are independent spawned processes (no shared lock): with a fixed temp name the second
``replace()`` finds the file gone and raises ``FileNotFoundError``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from src.utils.atomic_write import atomic_write_json


PROCS = 4
WRITES = 1000


def _hammer(target: str, n: int, queue: mp.Queue) -> None:
    errors = 0
    for i in range(n):
        try:
            atomic_write_json(Path(target), {'status': 'running', 'i': i, 'pid': os.getpid()})
        except FileNotFoundError:
            errors += 1
    queue.put(errors)


def _run(target: Path, procs: int) -> list[int]:
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    workers = [ctx.Process(target=_hammer, args=(str(target), WRITES, queue)) for _ in range(procs)]
    for w in workers:
        w.start()
    results = [queue.get(timeout=120) for _ in workers]
    for w in workers:
        w.join(timeout=30)
    return results


def test_concurrent_writers_raise_nothing(tmp_path: Path) -> None:
    target = tmp_path / 'job' / 'state.json'
    target.parent.mkdir()
    assert _run(target, PROCS) == [0] * PROCS
    assert json.loads(target.read_text())['status'] == 'running'
    assert [p.name for p in target.parent.iterdir()] == ['state.json']


def test_control_single_writer_round_trips(tmp_path: Path) -> None:
    target = tmp_path / 'job' / 'state.json'
    target.parent.mkdir()
    assert _run(target, 1) == [0]
    assert json.loads(target.read_text())['i'] == WRITES - 1


def test_atomic_write_json_leaves_no_temp_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / 'x.json'
    atomic_write_json(target, {'a': 1})
    with pytest.raises(TypeError):
        atomic_write_json(target, {'a': object()}, default=None)
    assert target.read_text() == '{"a": 1}'
    assert [p.name for p in tmp_path.iterdir()] == ['x.json']
