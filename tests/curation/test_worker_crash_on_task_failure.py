"""A worker stage task that dies outside its per-item try makes the process exit non-zero."""

from __future__ import annotations

import asyncio

import pytest

from scripts.curation.worker import crash


@pytest.fixture
def exits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    codes: list[int] = []
    monkeypatch.setattr(crash.os, '_exit', codes.append)
    return codes


@pytest.mark.asyncio
async def test_a_stage_that_raises_exits_nonzero(exits: list[int]) -> None:
    async def stage() -> None:
        raise RuntimeError('boom outside the per-item try')

    task = crash.spawn_critical(stage(), name='stage-a-0')
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert exits == [1]


@pytest.mark.asyncio
async def test_a_clean_drain_does_not_exit(exits: list[int]) -> None:
    async def stage() -> None:
        return None

    await crash.spawn_critical(stage(), name='stage-a-0')
    assert exits == []


@pytest.mark.asyncio
async def test_a_cancelled_stage_does_not_exit(exits: list[int]) -> None:
    async def stage() -> None:
        await asyncio.sleep(10)

    task = crash.spawn_critical(stage(), name='stage-a-0')
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert exits == []


def test_the_runner_spawns_its_stage_tasks_through_the_crash_wrapper() -> None:
    import inspect

    from scripts.curation.worker import runner

    source = inspect.getsource(runner)
    assert 'spawn_critical(stage_a_consumer' in source
    assert 'spawn_critical(writer' in source
    assert 'create_task(stage_' not in source
