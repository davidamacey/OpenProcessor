"""Interrupt, reconcile, resume and cancel of a combine, and undo (deleting
the target leaves the sources whole)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from src.services.projects.combine import execute

from .test_combine_execute import MAPPING, build
from .world import World, run_job, snapshot_indexes, tree_hash


def origins(world: World, slug: str) -> list[tuple[str, str]]:
    return sorted((d['origin_project'], d['origin_item_id']) for d in world.items(slug).values())


class Killed(BaseException):
    pass


def renamed(request: Any, slug: str) -> Any:
    return request.model_copy(update={'target': request.target.model_copy(update={'slug': slug})})


@pytest.mark.asyncio
async def test_cancel_after_a_chunk_settles_the_target_and_a_new_attempt_resumes_from_the_checkpoint(
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
    assert reference.status == 'completed'
    pages = len(calls)
    assert pages > 2
    calls.clear()
    world.settled.clear()

    run, _ = await run_job(world, request, cancel_after=1)
    assert run.status == 'cancelled'
    assert world.settled == [False]  # cancel is terminal: the target is failed, not left building
    assert 0 < len(world.items('combined')) < len(world.items('reference'))

    # The reaper resumes an interrupted job under the same id with the checkpoint it wrote.
    world.settled.clear()
    resumed, _ = await run_job(world, request, resume=run)
    assert resumed.status == 'completed'
    assert origins(world, 'combined') == origins(world, 'reference')
    assert len(origins(world, 'combined')) == len(set(origins(world, 'combined')))
    assert len(world.images('combined')) == len(world.images('reference'))
    assert len(calls) == pages  # the finished chunk was not redone


@pytest.mark.asyncio
async def test_an_error_mid_chunk_resumes_to_the_same_result(
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
            raise RuntimeError('worker died')  # the docs of this chunk are written, its mark is not
        return out

    monkeypatch.setattr(execute, 'copy_page', die_after_writing)
    run, _ = await run_job(world, request)
    assert run.status == 'failed'
    assert world.settled[-1] is False

    monkeypatch.setattr(execute, 'copy_page', real)
    resumed, _ = await run_job(world, request, resume=run)
    assert resumed.status == 'completed'
    assert origins(world, 'combined') == origins(world, 'reference')
    assert len(world.images('combined')) == len(world.images('reference'))


@pytest.mark.asyncio
async def test_resume_uses_the_persisted_plan_not_a_fresh_one(world: World) -> None:
    build(world)
    request = world.request(['cars-a', 'cars-b'], MAPPING)
    run, _ = await run_job(world, request)
    plan = execute.load_plan(run.params, run.document)
    assert plan.mapping.target_classes == ['car', 'truck']
    assert plan.mapping.per_source['cars-b']['sedan'].class_name == 'car'
    assert plan.duplicates  # the dedup decision is part of the plan
    assert plan.request.target.slug == 'combined'


@pytest.mark.asyncio
async def test_a_run_that_lost_its_lease_writes_nothing_more(world: World) -> None:
    build(world)
    request = world.request(['cars-a', 'cars-b'], MAPPING)
    run, _ = await run_job(world, request, lease_lost=True)
    assert run.result is None
    assert world.items('combined') == {}
    assert world.settled == []  # the successor settles the target


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
