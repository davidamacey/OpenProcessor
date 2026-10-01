"""W5: the bounds on a test run -- per-process concurrency per kind and a
wall-clock limit (the routes map them to 429 / 504)."""

from __future__ import annotations

import asyncio

import pytest

from src.services.curation import config_test_limits as limits


def test_the_over_cap_run_is_refused_and_a_released_slot_is_reusable() -> None:
    with limits.reserve_slot('vlm'), limits.reserve_slot('vlm'):
        assert limits.active_runs('vlm') == limits.SLOT_CAPS['vlm'] == 2
        with pytest.raises(limits.ConfigTestBusyError), limits.reserve_slot('vlm'):
            pytest.fail('a third VLM run must not start')
        assert limits.active_runs('vlm') == 2, 'a refused run holds no slot'
    assert limits.active_runs('vlm') == 0
    with limits.reserve_slot('vlm'):
        assert limits.active_runs('vlm') == 1


def test_the_kinds_are_capped_separately() -> None:
    with (
        limits.reserve_slot('vlm'),
        limits.reserve_slot('vlm'),
        limits.reserve_slot('segmenter'),
    ):
        assert limits.active_runs('segmenter') == 1


def test_a_slot_is_released_when_the_body_raises() -> None:
    with pytest.raises(RuntimeError, match='boom'), limits.reserve_slot('segmenter'):
        raise RuntimeError('boom')

    assert limits.active_runs('segmenter') == 0


@pytest.mark.asyncio
async def test_a_run_over_the_time_limit_is_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limits, 'TEST_TIMEOUT_S', 0.02)

    async def slow() -> str:
        await asyncio.sleep(5)
        return 'done'

    async def quick() -> str:
        return 'done'

    with pytest.raises(limits.ConfigTestTimeoutError):
        await limits.bounded(slow())
    assert await limits.bounded(quick()) == 'done'
