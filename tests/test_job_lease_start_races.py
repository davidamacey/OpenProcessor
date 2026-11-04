"""Eight spawned processes start the same singleton job at a barrier: exactly one starts.

The OpenSearch lease store is emulated by ``FakeLeaseOpenSearch`` over a manager dict (test
scaffolding for the server's per-document atomicity). The job state directories are real and
shared, as the ``/jobs`` volume is between ``api`` workers. The background body of each
family is replaced by a sleep so the started job stays live while the siblings try.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from typing import TYPE_CHECKING, Any

import pytest
from _fake_lease_opensearch import FakeLeaseOpenSearch


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

pytestmark = pytest.mark.real_lease_io

PROCS = 8


async def _idle(*_: object, **__: object) -> None:
    await asyncio.sleep(60)


async def _start_item_scores(client: Any) -> str:
    from src.services.curation.item_scores import job

    job.run_scoring_job = _idle  # type: ignore[assignment]
    try:
        await job.start_job(client, ['x'])
    except RuntimeError:
        return 'busy'
    return 'started'


STARTERS: dict[str, tuple[Callable[[Any], Awaitable[str]], dict[str, str]]] = {
    'item_scores': (_start_item_scores, {'OP_SCORES_STATE_DIR': 'scores'}),
}


async def _attempt(name: str, client: Any, barrier: Any, done: Any, out: Any) -> None:
    starter, _env = STARTERS[name]
    await asyncio.to_thread(barrier.wait, 120)
    out.put(await starter(client))
    await asyncio.to_thread(done.wait, 120)  # the winner's job stays live until all reported


def _child(name: str, store: Any, lock: Any, barrier: Any, done: Any, out: Any) -> None:
    from conftest import DEFAULT_RECORD

    from src.config.project_context import bind_process_project

    bind_process_project(DEFAULT_RECORD)  # type: ignore[arg-type]
    asyncio.run(_attempt(name, FakeLeaseOpenSearch(store, lock), barrier, done, out))


@pytest.mark.parametrize('name', sorted(STARTERS))
def test_exactly_one_of_eight_processes_starts_the_job(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var, sub in STARTERS[name][1].items():
        monkeypatch.setenv(var, str(tmp_path / sub))
    ctx = mp.get_context('spawn')
    with ctx.Manager() as manager:
        store, lock = manager.dict(), ctx.Lock()
        barrier, done, out = ctx.Barrier(PROCS), ctx.Event(), ctx.Queue()
        procs = [
            ctx.Process(target=_child, args=(name, store, lock, barrier, done, out))
            for _ in range(PROCS)
        ]
        for p in procs:
            p.start()
        outcomes = [out.get(timeout=180) for _ in range(PROCS)]
        done.set()
        for p in procs:
            p.join(timeout=60)

    assert sorted(outcomes) == ['busy'] * (PROCS - 1) + ['started'], os.getpid()
