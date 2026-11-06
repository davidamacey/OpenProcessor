"""The server-computed drain-stability verdict.

The ``(streak, zero_since)`` pair is the project's ``region_drain`` marker, so the verdict is
correct regardless of which ``api --workers`` process answers a given poll.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from src.db.repos.records import get_project_state
from src.db.session import async_session_factory, project_tx
from src.services.curation import region_drain


pytest_plugins = ['_records_world']
pytestmark = pytest.mark.db

T0 = datetime(2026, 1, 1, tzinfo=UTC)


async def _stored(world):
    async with async_session_factory()() as s, project_tx(s, world):
        row = await get_project_state(s, world, 'region_drain')
    return None if row is None else row.value


@pytest.mark.asyncio
async def test_nonzero_total_is_never_drained(session) -> None:
    v = await region_drain.observe_drain(session, 5, now=T0)
    assert v.drained is False
    assert v.stable_for_s == 0.0


@pytest.mark.asyncio
async def test_a_single_zero_reading_is_not_yet_drained(session) -> None:
    """A single zero right after a burst finishes could be a race, not a
    truly drained queue -- only stable_polls consecutive zeros count."""
    assert (await region_drain.observe_drain(session, 0, now=T0)).drained is False


@pytest.mark.asyncio
async def test_drained_after_stable_polls_consecutive_zeros(session) -> None:
    polls = region_drain.region_drain_stable_polls()
    for i in range(polls - 1):
        v = await region_drain.observe_drain(session, 0, now=T0 + timedelta(seconds=i))
        assert v.drained is False
    v = await region_drain.observe_drain(session, 0, now=T0 + timedelta(seconds=polls))
    assert v.drained is True
    assert v.stable_for_s > 0


@pytest.mark.asyncio
async def test_a_nonzero_reading_resets_the_streak(session, world) -> None:
    for i in range(region_drain.region_drain_stable_polls()):
        await region_drain.observe_drain(session, 0, now=T0 + timedelta(seconds=i))
    v = await region_drain.observe_drain(session, 3, now=T0 + timedelta(seconds=100))
    assert (v.drained, v.stable_for_s) == (False, 0.0)
    assert await _stored(world) == {'streak': 0, 'zero_since': None}
    # a fresh zero starts a new streak instead of resuming the old one
    assert (
        await region_drain.observe_drain(session, 0, now=T0 + timedelta(seconds=101))
    ).drained is False


@pytest.mark.asyncio
async def test_the_streak_is_shared_by_separate_sessions(world) -> None:
    """Polls landing on different worker processes see one streak (the bug the marker fixed)."""
    polls = region_drain.region_drain_stable_polls()
    verdict = None
    for i in range(polls):
        async with async_session_factory()() as other:  # a different "process" each poll
            verdict = await region_drain.observe_drain(other, 0, now=T0 + timedelta(seconds=i))
    assert verdict is not None
    assert verdict.drained is True
    assert (await _stored(world))['streak'] == polls


@pytest.mark.asyncio
async def test_stable_for_s_grows_across_sessions(world) -> None:
    async def poll(seconds: int) -> float:
        async with async_session_factory()() as s:
            return (
                await region_drain.observe_drain(s, 0, now=T0 + timedelta(seconds=seconds))
            ).stable_for_s

    assert await poll(0) == 0.0
    assert await poll(10) == pytest.approx(10.0)
    assert await poll(25) == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_simultaneous_polls_lose_no_increment(world) -> None:
    async def poll(i: int) -> None:
        async with async_session_factory()() as s:
            await region_drain.observe_drain(s, 0, now=T0 + timedelta(seconds=i))

    await asyncio.gather(*(poll(i) for i in range(6)))
    assert (await _stored(world))['streak'] == 6
