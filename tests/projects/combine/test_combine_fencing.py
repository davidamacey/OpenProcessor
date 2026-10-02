"""One live worker per combine job: a stalled worker that was taken over is
fenced off (in this process: resume refuses while it lives; across processes:
a claim it re-checks at every chunk boundary)."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from typing import Any

import pytest
from fastapi import HTTPException

from src.services.projects.combine import execute, service as svc
from src.services.projects.combine.store import iter_jobs

from .test_combine_execute import MAPPING, build
from .world import World, run_job


@pytest.mark.asyncio
async def test_a_stale_looking_job_whose_worker_is_alive_in_this_process_is_not_resumable(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(world)
    store, target = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    world.records[target.slug] = dataclasses.replace(target, status='building')
    live = peak = 0
    gate = asyncio.Event()

    async def parked_worker(_client: Any, **_kw: Any) -> bool:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        store.job.update(status='running')
        await gate.wait()
        live -= 1
        return False

    async def get_target(_client: Any, _slug: str) -> tuple[Any, int, int]:
        return world.records[target.slug], 1, 1

    monkeypatch.setattr(svc, 'run_combine', parked_worker)
    monkeypatch.setattr(svc, 'get_record_with_seq', get_target)
    store.job.update(status='interrupted')
    await svc.resume(world.fake, store.import_id)
    await asyncio.sleep(0.05)
    assert live == 1

    # The worker stalls past the stale window: a status read repairs the job
    # to interrupted, which a second resume would otherwise claim.
    old = time.time() - 600
    os.utime(store.job.heartbeat_file, (old, old))
    assert svc.job_state(store.import_id)['status'] == 'interrupted'
    with pytest.raises(HTTPException) as refused:
        await svc.resume(world.fake, store.import_id)
    assert refused.value.status_code == 409
    assert refused.value.detail['error'] == 'combine_not_resumable'  # type: ignore[index]

    await asyncio.sleep(0.05)
    assert (live, peak) == (1, 1)
    gate.set()
    await asyncio.sleep(0.05)


def _combined_store() -> Any:
    return next(s for s in iter_jobs() if s.job.read().get('target') == 'combined')


async def _taken_over_while_parked(
    world: World, monkeypatch: pytest.MonkeyPatch, *, park_in: str
) -> tuple[Any, int, tuple[dict[str, Any], dict[Any, Any]]]:
    """Worker A parks (in its first chunk's ``copy_page``, between its
    first and second page, after its last chunk, or inside the holdout step), worker B takes the job over and finishes it, then
    A is released. Returns the store, how many pages A copied, and the job
    state and chunk marks as B left them."""
    build(world)
    monkeypatch.setenv('OP_COMBINE_PAGE_SIZE', '2')
    request = world.request(['cars-a', 'cars-b'], MAPPING)
    entered, release = asyncio.Event(), asyncio.Event()
    a_pages = 0
    workers = 0
    real_copy = execute.copy_page
    real_pages = execute.iter_source_pages
    real_copy_all = execute._copy_all

    async def copy_page(ctx: Any, source_index: int, page: Any) -> Any:
        nonlocal a_pages
        if asyncio.current_task() is worker_a:
            a_pages += 1
            if park_in == 'copy_page' and a_pages == 1:
                entered.set()
                await release.wait()
        return await real_copy(ctx, source_index, page)

    async def copy_all(*args: Any) -> bool:
        cancelled = await real_copy_all(*args)
        if park_in == 'before_finish' and asyncio.current_task() is worker_a:
            entered.set()
            await release.wait()
        return cancelled

    def pages(*args: Any, **kwargs: Any) -> Any:
        nonlocal workers
        workers += 1
        is_a = workers == 1

        async def gen() -> Any:
            count = 0
            async for page in real_pages(*args, **kwargs):
                if is_a and park_in == 'between_pages' and count == 1:
                    entered.set()
                    await release.wait()
                count += 1
                yield page

        return gen()

    def parked_in_holdout(real: Any) -> Any:
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            out = await real(*args, **kwargs)
            if park_in == 'in_finish' and asyncio.current_task() is worker_a:
                entered.set()
                await release.wait()
            return out

        return wrapped

    monkeypatch.setattr(
        execute.holdout, 'record_union', parked_in_holdout(execute.holdout.record_union)
    )
    monkeypatch.setattr(execute.holdout, 'recompute', parked_in_holdout(execute.holdout.recompute))
    monkeypatch.setattr(execute, 'copy_page', copy_page)
    monkeypatch.setattr(execute, 'iter_source_pages', pages)
    monkeypatch.setattr(execute, '_copy_all', copy_all)
    worker_a = asyncio.create_task(run_job(world, request))
    await asyncio.wait_for(entered.wait(), 10)
    store = _combined_store()
    await run_job(world, request, resume=store)
    assert store.job.read()['status'] == 'completed'
    left = (store.job.read(), store.chunks_done())
    release.set()
    await worker_a
    return store, a_pages, left


@pytest.mark.asyncio
async def test_a_fenced_worker_mid_chunk_writes_no_mark_and_no_state(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _a_pages, (state, marks) = await _taken_over_while_parked(
        world, monkeypatch, park_in='copy_page'
    )
    assert store.job.read() == state
    assert store.chunks_done() == marks


@pytest.mark.asyncio
async def test_a_fenced_worker_between_chunks_copies_no_further_page(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, a_pages, (state, marks) = await _taken_over_while_parked(
        world, monkeypatch, park_in='between_pages'
    )
    assert a_pages == 1  # the page A had finished; none after it was taken over
    assert store.job.read() == state
    assert store.chunks_done() == marks


@pytest.mark.asyncio
async def test_a_fenced_worker_after_its_last_chunk_does_not_finish_the_job(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _a_pages, (state, marks) = await _taken_over_while_parked(
        world, monkeypatch, park_in='before_finish'
    )
    assert store.job.read() == state
    assert store.chunks_done() == marks


@pytest.mark.asyncio
async def test_a_fenced_worker_inside_the_holdout_step_does_not_write_completed(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _a_pages, (state, marks) = await _taken_over_while_parked(
        world, monkeypatch, park_in='in_finish'
    )
    assert store.job.read() == state
    assert store.chunks_done() == marks


@pytest.mark.asyncio
@pytest.mark.parametrize(('owned', 'settled'), [(True, 1), (False, 0)])
async def test_only_the_worker_that_owns_the_job_settles_the_target(
    world: World, monkeypatch: pytest.MonkeyPatch, owned: bool, settled: int
) -> None:
    build(world)
    store, target = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    assert store.job.read()['status'] == 'completed'
    finished: list[bool] = []

    async def finish_building(_client: Any, _target: Any, *, ok: bool) -> None:
        finished.append(ok)

    async def worker(_client: Any, **kw: Any) -> bool:
        if owned:
            await kw['settle'](True)
        return owned

    monkeypatch.setattr(svc, 'run_combine', worker)
    monkeypatch.setattr(svc.lifecycle, 'finish_building', finish_building)
    await svc._run(world.fake, store, [], target)
    assert len(finished) == settled
