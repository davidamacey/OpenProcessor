"""Interrupt, reconcile, resume and cancel of a combine, and undo (deleting
the target leaves the sources whole)."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from src.services.projects.combine import execute
from src.services.projects.combine.store import (
    ACTIVE_STATUSES,
    iter_jobs,
    open_job,
    reconcile_orphaned_jobs,
    running_jobs_for,
)

from .test_combine_execute import MAPPING, build
from .world import World, run_job, snapshot_indexes, tree_hash


def origins(world: World, slug: str) -> list[tuple[str, str]]:
    return sorted((d['origin_project'], d['origin_item_id']) for d in world.items(slug).values())


class Killed(BaseException):
    pass


def renamed(request: Any, slug: str) -> Any:
    return request.model_copy(update={'target': request.target.model_copy(update={'slug': slug})})


def combined_store() -> Any:
    return next(s for s in iter_jobs() if s.job.read().get('target') == 'combined')


@pytest.mark.asyncio
async def test_cancel_after_a_chunk_then_resume_finishes_without_duplicates(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(world)
    monkeypatch.setenv('OP_COMBINE_PAGE_SIZE', '2')
    request = world.request(['cars-a', 'cars-b'], MAPPING)
    real = execute.copy_page
    calls: list[int] = []

    async def counting(ctx: Any, source_index: int, page: Any) -> Any:
        calls.append(len(page))
        return await real(ctx, source_index, page)

    monkeypatch.setattr(execute, 'copy_page', counting)
    reference, _ = await run_job(world, renamed(request, 'reference'))
    assert reference.job.read()['status'] == 'completed'
    pages = len(calls)
    assert pages > 2
    calls.clear()

    async def cancel_after_first(ctx: Any, source_index: int, page: Any) -> Any:
        out = await counting(ctx, source_index, page)
        if len(calls) == 1:
            combined_store().job.request_cancel()
        return out

    monkeypatch.setattr(execute, 'copy_page', cancel_after_first)
    store, _ = await run_job(world, request)
    assert store.job.read()['status'] == 'cancelled'
    assert 0 < len(world.items('combined')) < len(world.items('reference'))

    monkeypatch.setattr(execute, 'copy_page', counting)
    await run_job(world, request, resume=store)
    assert store.job.read()['status'] == 'completed'
    assert origins(world, 'combined') == origins(world, 'reference')
    assert len(origins(world, 'combined')) == len(set(origins(world, 'combined')))
    assert len(world.images('combined')) == len(world.images('reference'))
    assert len(calls) == pages  # the finished chunk was not redone


@pytest.mark.asyncio
async def test_a_kill_mid_chunk_is_repaired_and_resumes_to_the_same_result(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(world)
    monkeypatch.setenv('OP_COMBINE_PAGE_SIZE', '2')
    request = world.request(['cars-a', 'cars-b'], MAPPING)
    await run_job(world, renamed(request, 'reference'))

    real = execute.copy_page
    seen = {'n': 0}

    async def die_after_writing(ctx: Any, source_index: int, page: Any) -> Any:
        out = await real(ctx, source_index, page)
        seen['n'] += 1
        if seen['n'] == 2:
            raise Killed  # the docs of this chunk are written, its mark is not
        return out

    monkeypatch.setattr(execute, 'copy_page', die_after_writing)
    with pytest.raises(Killed):
        await run_job(world, request)
    store = combined_store()
    assert store.job.read()['status'] == 'running'  # nothing got to finish it

    # Startup repair: a dead process's live-looking job becomes interrupted.
    old = time.time() - 600
    os.utime(store.job.heartbeat_file, (old, old))
    assert reconcile_orphaned_jobs() == 1
    assert store.job.read()['status'] == 'interrupted'
    assert not running_jobs_for('cars-a')  # no longer busy

    monkeypatch.setattr(execute, 'copy_page', real)
    # A cold worker: nothing but the files on disk.
    cold = open_job(store.import_id)
    assert cold is not None
    await run_job(world, request, resume=cold)
    assert cold.job.read()['status'] == 'completed'
    assert origins(world, 'combined') == origins(world, 'reference')
    assert len(world.images('combined')) == len(world.images('reference'))


@pytest.mark.asyncio
async def test_resume_uses_the_persisted_plan_not_a_fresh_one(world: World) -> None:
    build(world)
    request = world.request(['cars-a', 'cars-b'], MAPPING)
    store, _ = await run_job(world, request)
    plan = execute.load_plan(store)
    assert plan.mapping.target_classes == ['car', 'truck']
    assert plan.mapping.per_source['cars-b']['sedan'].class_name == 'car'
    assert plan.duplicates  # the dedup decision is part of the plan
    assert plan.request.target.slug == 'combined'


@pytest.mark.asyncio
async def test_a_job_with_a_live_heartbeat_marks_its_projects_busy(world: World) -> None:
    build(world)
    request = world.request(['cars-a', 'cars-b'], MAPPING)
    store, _ = await run_job(world, request)
    store.job.update(status='running')
    store.job.touch_heartbeat()
    assert store.job.read()['status'] in ACTIVE_STATUSES
    assert [j for j, _ in running_jobs_for('cars-a')] == [store.import_id]
    assert [j for j, _ in running_jobs_for('combined')] == [store.import_id]
    assert not running_jobs_for('unrelated')


@pytest.mark.asyncio
async def test_deleting_the_target_leaves_the_sources_whole(world: World) -> None:
    build(world)
    slugs = ['cars-a', 'cars-b']
    before = snapshot_indexes(world, slugs)
    trees = {s: tree_hash(world.records[s].resources.upload_root) for s in slugs}
    await run_job(world, world.request(slugs, MAPPING))
    # What a project delete removes: the target's indexes and its upload dir.
    for index in (world.items_index('combined'), world.images_index('combined')):
        world.fake.store.pop(index, None)
    shutil.rmtree(world.records['combined'].resources.upload_root)
    assert snapshot_indexes(world, slugs) == before
    assert {s: tree_hash(world.records[s].resources.upload_root) for s in slugs} == trees
    for image in world.images('cars-a').values():
        assert Path(image['image_path']).read_bytes()  # hard links never shared the source's fate


@pytest.mark.asyncio
async def test_progress_events_name_the_target_and_busy_lists_the_job(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import event_hub
    from src.services.projects.busy import _combine_jobs

    build(world)
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        event_hub,
        'publish_global_event',
        lambda event_type, **fields: events.append((event_type, fields)),
    )
    store, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    assert events
    assert {t for t, _ in events} == {'combine.progress'}
    assert all(f['target'] == 'combined' and f['job_id'] == store.import_id for _, f in events)
    assert events[-1][1]['status'] == 'completed'

    store.job.update(status='running')
    store.job.touch_heartbeat()
    busy = _combine_jobs(world.records['cars-a'])
    assert [(j.kind, j.job_id) for j in busy] == [('combine', store.import_id)]
