# ruff: noqa: F811 - pytest fixtures are imported by name and used as test arguments
"""Auto-label loops belong to one project: its run rows, and its hold on retraining."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from fixtures.job_fixtures import job_env, kinds, producer, wakes  # noqa: F401
from fixtures.job_rows import make_project

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import new_project_record
from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.schemas.curation.assist_loop import AssistLoopRequest, LoopBudget
from src.services.curation.assist_loop import service
from src.services.curation.assist_loop.partition import RootSize


if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.db, pytest.mark.asyncio]


def _record(pg: Any, slug: str) -> Any:
    return replace(new_project_record(slug, base_curation_config()), id=make_project(pg, slug))


@pytest.fixture
def records(pg: Any, job_env: Any) -> dict[str, Any]:
    tag = uuid.uuid4().hex[:6]
    return {name: _record(pg, f'{name}-{tag}') for name in ('alpha', 'beta')}


@pytest.fixture
def alpha_loop(records: dict[str, Any], producer: Any, wakes: Any) -> Iterator[str]:
    prepared = SimpleNamespace(
        request=AssistLoopRequest(dry_run=False, budget=LoopBudget(max_vlm_crops=100)),
        vlm=SimpleNamespace(name='vlm', revision=1, endpoint=SimpleNamespace(model_id='m')),
        audit_vlm=None,
        pack=SimpleNamespace(name='pack'),
        pack_revision=1,
        images_per_call=4,
        roots=[RootSize(str(uuid.uuid4()), 1, 300, 300)],
    )

    async def start() -> str:
        async with async_session_factory()() as session:
            return (await service.start_loop(session, prepared)).loop_id

    with bind_project(records['alpha']):
        yield asyncio.run(start())


async def test_a_loop_is_served_to_its_own_project_only(
    records: dict[str, Any], alpha_loop: str
) -> None:
    async with async_session_factory()() as session:
        with bind_project(records['alpha']):
            assert (await service.get_state(session, alpha_loop)).loop_id == alpha_loop
            listed = await service.list_states(session, active_only=False, limit=10)
            assert [x.loop_id for x in listed.loops] == [alpha_loop]
        with bind_project(records['beta']):
            assert (await service.list_states(session, active_only=False, limit=10)).loops == []
            for call in (service.get_state, service.cancel, service.pause, service.resume):
                with pytest.raises(RequestRejectedError) as refused:
                    await call(session, alpha_loop)
                assert refused.value.status_code == 404, call.__name__
        with bind_project(records['alpha']):
            assert (await service.get_state(session, alpha_loop)).status == 'queued', (
                'beta cancelled nothing'
            )


async def test_a_live_loop_holds_only_its_own_projects_retraining(
    records: dict[str, Any], alpha_loop: str
) -> None:
    async with async_session_factory()() as session:
        with bind_project(records['alpha']), pytest.raises(RequestRejectedError) as held:
            await service.require_no_live_loop(session)
        assert held.value.detail['error'] == 'assist_loop_running'
        assert held.value.detail['loop_id'] == alpha_loop
        with bind_project(records['beta']):
            await service.require_no_live_loop(session)
