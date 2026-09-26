"""P1 commit 1: bound-project context (see
docs/design/openprocessor_internal/projects_plan.md §10 P1 test list)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from src.config.curation import base_curation_config
from src.config.project_context import (
    ProjectNotBound,
    bind_project,
    current_project,
    project_env,
    run_in_executor_bound,
)
from src.config.projects import ProjectRecord, resources_for_default


# These tests are about binding itself: no autouse `default` binding.
pytestmark = pytest.mark.unbound


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_default(base_curation_config()),
    )


def test_unbound_raises() -> None:
    with pytest.raises(ProjectNotBound):
        current_project()


def test_bind_and_read() -> None:
    with bind_project(_record('alpha')):
        assert current_project().record.slug == 'alpha'
    with pytest.raises(ProjectNotBound):
        current_project()


def test_nesting_restores_outer_binding() -> None:
    with bind_project(_record('alpha')):
        assert current_project().record.slug == 'alpha'
        with bind_project(_record('beta')):
            assert current_project().record.slug == 'beta'
        assert current_project().record.slug == 'alpha'


def test_create_task_inherits_binding() -> None:
    async def _inner() -> str:
        return current_project().record.slug

    async def _run() -> str:
        with bind_project(_record('alpha')):
            task = asyncio.create_task(_inner())
            return await task

    assert asyncio.run(_run()) == 'alpha'


def test_to_thread_inherits_binding() -> None:
    def _inner() -> str:
        return current_project().record.slug

    async def _run() -> str:
        with bind_project(_record('alpha')):
            return await asyncio.to_thread(_inner)

    assert asyncio.run(_run()) == 'alpha'


def test_bare_run_in_executor_does_not_inherit_binding() -> None:
    """Documents the pitfall run_in_executor_bound exists to fix."""

    def _inner() -> str:
        return current_project().record.slug

    async def _run() -> str:
        with bind_project(_record('alpha')), ThreadPoolExecutor(max_workers=1) as pool:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(pool, _inner)

    with pytest.raises(ProjectNotBound):
        asyncio.run(_run())


def test_run_in_executor_bound_inherits_binding() -> None:
    def _inner() -> str:
        return current_project().record.slug

    async def _run() -> str:
        with bind_project(_record('alpha')), ThreadPoolExecutor(max_workers=1) as pool:
            loop = asyncio.get_running_loop()
            return await run_in_executor_bound(loop, pool, _inner)

    assert asyncio.run(_run()) == 'alpha'


def test_project_env() -> None:
    with bind_project(_record('alpha')):
        assert project_env() == {'OP_PROJECT': 'alpha'}


def test_two_sequential_testclient_requests_see_own_binding() -> None:
    """Mirrors the plan's TestClient scenario without needing the route
    mounting that lands in commit 4: each request-shaped call binds and
    unbinds independently via ``set_bound_project``, one call per
    (simulated) request task, exactly as a FastAPI dependency would."""
    from src.config.project_context import set_bound_project

    async def _handle(slug: str) -> str:
        set_bound_project(_record(slug))
        return current_project().record.slug

    async def _run() -> tuple[str, str]:
        first = await asyncio.create_task(_handle('alpha'))
        second = await asyncio.create_task(_handle('beta'))
        return first, second

    assert asyncio.run(_run()) == ('alpha', 'beta')
