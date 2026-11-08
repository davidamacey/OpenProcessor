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
import time
from typing import TYPE_CHECKING, Any

import pytest
from _fake_lease_opensearch import FakeLeaseOpenSearch


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

pytestmark = pytest.mark.real_lease_io

PROCS = 8


def _widen_check(module: Any, name: str = '_is_busy') -> None:
    """Hold every process between its busy check and its claim, as a slow request would, so
    all eight see an idle job and only the lease decides who starts."""
    real = getattr(module, name)

    def slow(*args: Any, **kwargs: Any) -> Any:
        busy = real(*args, **kwargs)
        time.sleep(0.5)
        return busy

    setattr(module, name, slow)


async def _idle(*_: object, **__: object) -> None:
    await asyncio.sleep(60)


async def _start_item_scores(client: Any) -> str:
    from src.services.curation.item_scores import job

    job.run_scoring_job = _idle  # type: ignore[assignment]
    _widen_check(job)
    try:
        await job.start_job(client, ['x'])
    except RuntimeError:
        return 'busy'
    return 'started'


async def _start_selection(client: Any) -> str:
    from src.services.curation.selection import job

    job.run_selection_job = _idle  # type: ignore[assignment]
    _widen_check(job)
    try:
        await job.start_job(client, index='i', query={}, k=1, seed_crop_id=None, scope={}, max_n=10)
    except RuntimeError:
        return 'busy'
    return 'started'


async def _start_embedding_viz(client: Any) -> str:
    from src.services.curation import embedding_viz

    embedding_viz.run_projection_job = _idle  # type: ignore[assignment]
    _widen_check(embedding_viz)
    try:
        await embedding_viz.start_job(client, scope='residual')
    except RuntimeError:
        return 'busy'
    return 'started'


async def _start_probe(client: Any) -> str:
    from pathlib import Path

    from src.services.curation import probe_job

    probe_job._run = _idle  # type: ignore[assignment]
    _widen_check(probe_job)
    try:
        await probe_job.start_probe_job('j', 't', Path('/x/best.pt'), client)
    except probe_job.ProbeJobBusyError:
        return 'busy'
    return 'started'


async def _start_reprocess(client: Any) -> str:
    from src.services.curation import reprocess_job

    _widen_check(reprocess_job, 'running_job_ids')
    try:
        await reprocess_job.create_job(client, request={}, scopes=['embed'], image_ids=['a'])
    except reprocess_job.ReprocessBusyError:
        return 'busy'
    return 'started'


async def _start_assist(client: Any) -> str:
    from src.core.exceptions import RequestRejectedError
    from src.services.curation.assist import job

    _widen_check(job, 'live_run_ids')
    try:
        await job.create_run(client, state={'status': 'queued'}, request={}, members=None)
    except RequestRejectedError:
        return 'busy'
    return 'started'


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
    'item_scores': (_start_item_scores, {'OP_SCORES_STATE_DIR': 'scores'}),
    'selection': (_start_selection, {'OP_SELECT_JOBS_DIR': 'selection'}),
    'embedding_viz': (_start_embedding_viz, {'OP_VIZ_JOBS_DIR': 'viz'}),
    'reprocess': (_start_reprocess, {'OP_REPROCESS_JOBS_DIR': 'reprocess'}),
    'assist_run': (_start_assist, {'OP_ASSIST_JOBS_DIR': 'assist'}),
    'probe': (_start_probe, {'OP_PROBE_JOBS_DIR': 'probe'}),
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
