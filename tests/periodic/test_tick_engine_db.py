"""Connections opened by a tick do not grow with its projects (needs the db tier).

The shape is the deepest a tick opens: the advisory-lock connection, a session inside it and a
second, nested session (a job start inside a project's pass).
"""

from __future__ import annotations

import pytest
from sqlalchemy import event, text

from src.db.engine import TICK_POOL_SIZE, current_async_engine
from src.db.session import async_session_factory
from src.services.periodic.runner import run_periodic, tick_lock


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_env')]


def test_five_projects_open_at_most_the_pool_size_of_connections() -> None:
    async def body() -> int:
        opened: list[int] = []
        event.listen(current_async_engine().sync_engine, 'connect', lambda *_: opened.append(1))
        for obj in range(5):
            async with tick_lock(901, obj) as held:
                assert held
                async with async_session_factory()() as outer, async_session_factory()() as inner:
                    await outer.execute(text('SELECT 1'))
                    await inner.execute(text('SELECT 1'))
        return len(opened)

    assert run_periodic('nesting', body()) <= TICK_POOL_SIZE
