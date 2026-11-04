"""Eight spawned processes race for one lease: exactly one wins, in every round.

The OpenSearch server is emulated by ``FakeLeaseOpenSearch`` over a ``multiprocessing``
manager dict (test scaffolding for the server's per-document atomicity, not production
locking). The workers are started once and race ``ROUNDS`` times at a barrier, each round for
a lease of its own. ``spawn`` so no forked child inherits a thread.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from typing import Any

import pytest
from _fake_lease_opensearch import FakeLeaseOpenSearch

from src.clients import job_lease
from src.domain.job_lease import Acquired, Busy


pytestmark = pytest.mark.real_lease_io

PROCS = 8
ROUNDS = 10


def _race(store: Any, lock: Any, barrier: Any, out: Any, kind: str) -> None:
    from conftest import DEFAULT_RECORD

    from src.config.project_context import bind_process_project

    bind_process_project(DEFAULT_RECORD)  # type: ignore[arg-type]
    client = FakeLeaseOpenSearch(store, lock)
    for round_no in range(ROUNDS):
        barrier.wait(timeout=120)
        result = asyncio.run(job_lease.acquire(client, kind, str(round_no), f'job-{os.getpid()}'))
        out.put((round_no, type(result).__name__))


@pytest.mark.parametrize('kind', ['reprocess', 'probe'])
def test_exactly_one_of_eight_processes_acquires_each_round(kind: str) -> None:
    ctx = mp.get_context('spawn')
    with ctx.Manager() as manager:
        store, lock = manager.dict(), ctx.Lock()
        barrier, out = ctx.Barrier(PROCS), ctx.Queue()
        procs = [
            ctx.Process(target=_race, args=(store, lock, barrier, out, kind)) for _ in range(PROCS)
        ]
        for p in procs:
            p.start()
        results = [out.get(timeout=180) for _ in range(PROCS * ROUNDS)]
        for p in procs:
            p.join(timeout=60)

    for round_no in range(ROUNDS):
        names = sorted(name for r, name in results if r == round_no)
        assert names == [Acquired.__name__] + [Busy.__name__] * (PROCS - 1), (round_no, names)
