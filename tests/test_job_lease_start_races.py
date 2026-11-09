"""Eight spawned processes start the same singleton job at a barrier: exactly one starts.

The OpenSearch lease store is emulated by ``FakeLeaseOpenSearch`` over a manager dict (test
scaffolding for the server's per-document atomicity). The background body of each
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


async def _start_region_job(client: Any, which: str) -> str:
    """The region jobs return a snapshot either way, so a run counts as started when its
    own body ran in this process."""
    from src.services.curation.clustering import region_cluster_jobs as jobs

    ran = asyncio.Event()

    async def body(*_: object, **__: object) -> dict[str, object]:
        ran.set()
        await asyncio.sleep(60)
        return {}

    jobs.build_region_fp_centroids = body  # type: ignore[assignment]
    jobs.cluster_region_residuals = body  # type: ignore[assignment]
    jobs.count_false_positive_boxes = body  # type: ignore[assignment]
    if which == 'region_cluster':
        await jobs.start_region_cluster_job(client, rebuild_fp_centroids=False, auto_fp_threshold=0)
    else:
        await jobs.start_region_fp_centroid_job(client)
    await asyncio.sleep(1.5)
    return 'started' if ran.is_set() else 'busy'


async def _start_region_cluster(client: Any) -> str:
    return await _start_region_job(client, 'region_cluster')


async def _start_region_fp_centroids(client: Any) -> str:
    return await _start_region_job(client, 'region_fp_centroids')


STARTERS: dict[str, tuple[Callable[[Any], Awaitable[str]], dict[str, str]]] = {
    'region_cluster': (_start_region_cluster, {'OP_STATE_DIR': 'state'}),
    'region_fp_centroids': (_start_region_fp_centroids, {'OP_STATE_DIR': 'state'}),
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
