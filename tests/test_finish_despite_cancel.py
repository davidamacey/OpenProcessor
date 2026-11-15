"""``finish_despite_cancel``: a step that must end (a stored photo's release and events) survives
cancellation of the task that awaits it, also a second one (Python docs, ``asyncio.shield``)."""

from __future__ import annotations

import asyncio

import pytest

from src.utils.task_group import finish_despite_cancel


async def _settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_step_runs_to_its_end_through_two_cancellations() -> None:
    gate = asyncio.Event()
    ended: list[str] = []

    async def step() -> str:
        await gate.wait()
        ended.append('done')
        return 'value'

    async def caller() -> str:
        return await finish_despite_cancel(step())

    task = asyncio.create_task(caller())
    await _settle()
    task.cancel()
    await _settle()
    task.cancel()  # the second cancellation of the same awaiting task
    await _settle()
    assert ended == []  # still running: neither cancellation reached it
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ended == ['done']


@pytest.mark.asyncio
async def test_without_a_cancellation_the_result_and_the_exception_pass_through() -> None:
    async def ok() -> int:
        return 7

    async def boom() -> int:
        raise ValueError('step failed')

    assert await finish_despite_cancel(ok()) == 7
    with pytest.raises(ValueError, match='step failed'):
        await finish_despite_cancel(boom())


@pytest.mark.asyncio
async def test_a_plain_await_in_a_finally_is_interrupted_by_a_second_cancellation() -> None:
    """The failure this guards against, shown on the unprotected form."""
    gate = asyncio.Event()
    ended: list[str] = []

    async def step() -> None:
        await gate.wait()
        ended.append('done')

    async def caller() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            await step()

    task = asyncio.create_task(caller())
    await _settle()
    task.cancel()
    await _settle()
    task.cancel()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ended == []
