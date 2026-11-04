"""Live: eight processes race for one job lease on the harness's real OpenSearch.

The offline race tests emulate the server's per-document atomicity; this one asks the real
server (``op_type=create`` and ``if_seq_no`` / ``if_primary_term``) and records the acquire
latency. The lease documents live in the harness's folded configs index and are deleted at
the end. Five repetitions in the B1 smoke check; ``LEASE_LIVE_REPS=20`` for a local run.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import statistics
import time
import uuid
from typing import Any

import pytest

from tests.live.conftest import OPENSEARCH_URL


pytestmark = pytest.mark.live

PROCS = 8
REPS = int(os.environ.get('LEASE_LIVE_REPS', '5'))
LEASE_INDEX = 'verify_prj_default__configs'  # the harness's folded SETTINGS role index


def _race(url: str, kind: str, barrier: Any, out: Any) -> None:
    from opensearchpy import AsyncOpenSearch

    from src.clients import job_lease

    job_lease.idx = lambda _role: LEASE_INDEX  # type: ignore[assignment]

    async def run() -> None:
        client = AsyncOpenSearch(hosts=[url], use_ssl=False)
        try:
            for rep in range(REPS):
                barrier.wait(timeout=120)
                started = time.perf_counter()
                result = await job_lease.acquire(client, kind, str(rep), f'job-{os.getpid()}')
                out.put((rep, type(result).__name__, time.perf_counter() - started))
        finally:
            await client.close()

    asyncio.run(run())


def _delete_leases(kind: str) -> None:
    from opensearchpy import AsyncOpenSearch

    async def run() -> None:
        client = AsyncOpenSearch(hosts=[OPENSEARCH_URL], use_ssl=False)
        try:
            for rep in range(REPS):
                await client.delete(index=LEASE_INDEX, id=f'lease:{kind}:{rep}', ignore=[404])
        finally:
            await client.close()

    asyncio.run(run())


def test_exactly_one_of_eight_processes_acquires_each_lease_on_the_real_server() -> None:
    kind = f'live_probe_{uuid.uuid4().hex[:8]}'
    ctx = mp.get_context('spawn')
    barrier, out = ctx.Barrier(PROCS), ctx.Queue()
    procs = [
        ctx.Process(target=_race, args=(OPENSEARCH_URL, kind, barrier, out)) for _ in range(PROCS)
    ]
    try:
        for p in procs:
            p.start()
        results = [out.get(timeout=300) for _ in range(PROCS * REPS)]
        for p in procs:
            p.join(timeout=60)
    finally:
        _delete_leases(kind)

    for rep in range(REPS):
        names = sorted(name for r, name, _ in results if r == rep)
        assert names == ['Acquired'] + ['Busy'] * (PROCS - 1), (rep, names)
    latencies = sorted(t * 1000 for _, _, t in results)
    p50 = statistics.median(latencies)
    p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]
    print(f'lease acquire latency over {len(latencies)} calls: p50={p50:.1f} ms p99={p99:.1f} ms')


def test_control_a_sequential_acquire_and_release_on_the_real_server() -> None:
    from opensearchpy import AsyncOpenSearch

    from src.clients import job_lease
    from src.domain.job_lease import Acquired

    job_lease.idx = lambda _role: LEASE_INDEX  # type: ignore[assignment]
    kind = f'live_seq_{uuid.uuid4().hex[:8]}'

    async def run() -> None:
        client = AsyncOpenSearch(hosts=[OPENSEARCH_URL], use_ssl=False)
        try:
            first = await job_lease.acquire(client, kind, None, 'a')
            assert isinstance(first, Acquired)
            assert await job_lease.release(client, first.lease)
            second = await job_lease.acquire(client, kind, None, 'b')
            assert isinstance(second, Acquired)
            assert second.lease.generation == 2
        finally:
            await client.delete(index=LEASE_INDEX, id=f'lease:{kind}', ignore=[404])
            await client.close()

    asyncio.run(run())
