"""The detection worker idles while a GPU claim or a model swap is recorded, and resumes after."""

from __future__ import annotations

import asyncio

import pytest

from scripts.curation.worker import state


@pytest.mark.asyncio
async def test_the_worker_does_not_wait_when_nothing_is_claimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def free() -> bool:
        return False

    monkeypatch.setattr(state, 'vlm_paused', free)

    await asyncio.wait_for(state._wait_while_gpu_claimed(sleep_s=0.01), 1)


@pytest.mark.asyncio
async def test_the_worker_polls_while_claimed_and_resumes_when_it_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = [True, True, False]
    sleeps: list[float] = []

    async def paused() -> bool:
        return answers.pop(0)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(state, 'vlm_paused', paused)
    monkeypatch.setattr(state.asyncio, 'sleep', sleep)

    await state._wait_while_gpu_claimed(sleep_s=3.0)

    assert sleeps == [3.0, 3.0]
    assert answers == []
