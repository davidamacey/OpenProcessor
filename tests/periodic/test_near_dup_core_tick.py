"""The core grouping tick: one ``near_dup_core_group`` job while core pending rows exist, none
when there is nothing to group, and never two live at once (the exclusive group)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fixtures.job_rows import sql

from src.db.session import async_session_factory, core_tx
from src.services.near_duplicates import pending
from src.services.periodic import near_dup_ticks
from src.services.periodic.runner import run_periodic


pytestmark = [
    pytest.mark.db,
    pytest.mark.usefixtures('pg_registry', 'job_env', 'producer', 'wakes'),
]

KIND = 'near_dup_core_group'


def _mark_core_pending(ids: list[str]) -> None:
    async def go() -> None:
        async with async_session_factory()() as session, core_tx(session, 'test pending'):
            await pending.enqueue_core_pending(session, ids, 'added')

    asyncio.run(go())


def _jobs(pg: Any) -> list[tuple[Any, ...]]:
    return sql(pg, 'SELECT project_id, params FROM jobs WHERE kind = %s', KIND)


def _tick(now: float) -> Any:
    return run_periodic('near_dup_core_group', near_dup_ticks.core_group_tick(now=now))


def test_nothing_pending_starts_nothing(pg) -> None:
    report = _tick(1_800_000_000.0)

    assert _jobs(pg) == []
    assert report.done == ['core']


def test_pending_rows_start_one_deployment_wide_job_and_a_live_one_blocks_the_next(pg) -> None:
    _mark_core_pending(['a', 'b'])

    _tick(1_800_000_000.0)
    _tick(1_800_000_000.0)  # the same slot delivered twice
    report = _tick(1_800_000_060.0)  # a later slot, but the first job is still live

    assert _jobs(pg) == [(None, {})]
    assert report.skipped_busy == ['core']
