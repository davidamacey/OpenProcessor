"""``service.resume`` takes a claim (one worker per job), applies the same
source gate as preview and start, and never blocks the event loop."""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from typing import Any

import pytest
from fastapi import HTTPException

from src.services.projects.combine import image_copy, plan, service as svc

from .test_combine_execute import MAPPING, build
from .world import World, run_job


async def _interrupted(world: World, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
    build(world)
    request = world.request(['cars-a', 'cars-b'], MAPPING)
    store, target = await run_job(world, request)
    store.job.update(status='interrupted')
    world.records[target.slug] = dataclasses.replace(target, status='building')
    spawned: list[Any] = []

    async def get_target(_client: Any, _slug: str) -> tuple[Any, int, int]:
        await asyncio.sleep(0.01)  # a real OpenSearch round trip
        return world.records[target.slug], 1, 1

    monkeypatch.setattr(svc, 'get_record_with_seq', get_target)
    monkeypatch.setattr(svc, '_spawn', lambda *a, **_k: spawned.append(a))
    return store, spawned


@pytest.mark.asyncio
async def test_two_concurrent_resumes_spawn_exactly_one_worker(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, spawned = await _interrupted(world, monkeypatch)
    results = await asyncio.gather(
        svc.resume(world.fake, store.import_id),
        svc.resume(world.fake, store.import_id),
        return_exceptions=True,
    )
    assert len(spawned) == 1
    assert sum(isinstance(r, Exception) for r in results) == 1


@pytest.mark.asyncio
async def test_a_resume_of_a_job_that_was_already_resumed_is_refused(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, spawned = await _interrupted(world, monkeypatch)
    await svc.resume(world.fake, store.import_id)
    with pytest.raises(HTTPException) as refused:
        await svc.resume(world.fake, store.import_id)
    assert refused.value.status_code == 409
    assert len(spawned) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', ['deleted', 'deleting', 'failed', 'building'])
async def test_resume_refuses_a_source_that_preview_refuses(
    world: World, monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    store, spawned = await _interrupted(world, monkeypatch)
    world.records['cars-b'] = dataclasses.replace(world.records['cars-b'], status=bad)
    with pytest.raises(HTTPException) as refused:
        await svc.resume(world.fake, store.import_id)
    assert refused.value.status_code == 409
    assert not spawned
    assert store.job.read()['status'] == 'interrupted'


@pytest.mark.asyncio
async def test_resume_refuses_a_source_busy_with_another_job(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.projects import busy

    store, spawned = await _interrupted(world, monkeypatch)
    other = busy.JobRef(kind='dataset_import', job_id='imp_x')
    monkeypatch.setattr(busy, 'running_jobs', lambda _record: [other])
    with pytest.raises(HTTPException) as refused:
        await svc.resume(world.fake, store.import_id)
    assert refused.value.status_code == 409
    assert not spawned


@pytest.mark.asyncio
async def test_a_job_is_not_busy_with_itself_on_resume(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.projects import busy

    store, spawned = await _interrupted(world, monkeypatch)
    me = busy.JobRef(kind='combine', job_id=store.import_id)
    monkeypatch.setattr(busy, 'running_jobs', lambda _record: [me])
    await svc.resume(world.fake, store.import_id)
    assert len(spawned) == 1


@pytest.mark.asyncio
async def test_blocking_file_work_runs_off_the_event_loop(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(world)
    main = threading.get_ident()
    seen: dict[str, int] = {}

    def spy(name: str, real: Any) -> Any:
        def wrapper(*a: Any, **k: Any) -> Any:
            seen[name] = threading.get_ident()
            return real(*a, **k)

        return wrapper

    monkeypatch.setattr(image_copy, 'link_or_copy', spy('copy', image_copy.link_or_copy))
    monkeypatch.setattr(plan, 'find_duplicates', spy('hash', plan.find_duplicates))
    await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    assert seen.keys() == {'copy', 'hash'}
    assert main not in seen.values()


@pytest.mark.asyncio
async def test_resume_refuses_when_a_source_changed_since_the_preview(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, spawned = await _interrupted(world, monkeypatch)
    world.add_image('cars-a', items=[{'cls': 'car', 'bbox': [0.1, 0.1, 0.4, 0.4]}])
    with pytest.raises(HTTPException) as refused:
        await svc.resume(world.fake, store.import_id)
    assert refused.value.status_code == 409
    assert refused.value.detail['error'] == 'preview_stale'  # type: ignore[index]
    assert not spawned
    assert store.job.read()['status'] == 'interrupted'
