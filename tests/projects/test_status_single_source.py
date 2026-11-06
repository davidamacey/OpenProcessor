"""The route guard and GET /projects/{slug} read one status function.

The registry revision counter is bumped after the project doc is written, so
a worker whose snapshot still says ``building`` must confirm it at the doc:
at every instant of the flip window the guard and the GET must agree.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import aseed_project, seed_default_project


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]


async def _guard(slug: str) -> str:
    from src.routers.curation._project_deps import _resolve_and_bind

    try:
        return (await _resolve_and_bind(slug)).status
    except HTTPException as exc:
        return exc.detail['error']


async def _get_route(slug: str) -> str:
    from src.routers.curation import projects as routes

    try:
        resp = await routes.get_project(slug)
    except HTTPException as exc:
        return exc.detail['error']
    return resp.status


def test_guard_and_get_agree_in_the_flip_window(monkeypatch) -> None:
    from src.routers.curation import projects as routes

    async def _no_os():
        raise RuntimeError('no opensearch in this test')

    monkeypatch.setattr(routes, 'make_curation_opensearch', _no_os)

    async def _run() -> list[tuple[str, str]]:
        await seed_default_project()
        await aseed_project('combo', status='building')
        registry = ProjectRegistry()
        set_project_registry(registry)
        await registry.ensure_fresh()
        seen = [(await _guard('combo'), await _get_route('combo'))]
        assert seen[0] == ('project_building', 'building')
        # The flip: the row is active, the revision counter not bumped yet.
        await _set_status_without_bump('combo', 'active')
        for _ in range(3):
            seen += [(await _guard('combo'), await _get_route('combo'))]
        return seen

    seen = asyncio.run(_run())
    for guard, get in seen[1:]:
        assert (guard, get) == ('active', 'active')


async def _set_status_without_bump(slug: str, status: str) -> None:
    from sqlalchemy import update

    from src.db.models.projects import Project
    from src.services.projects.store import run_repo

    async def _set(session):
        await session.execute(update(Project).where(Project.slug == slug).values(status=status))

    await run_repo('flip a status without the counter', _set)


def test_unreachable_row_keeps_snapshot(monkeypatch) -> None:
    async def _run() -> str:
        await seed_default_project()
        await aseed_project('combo', status='building')
        registry = ProjectRegistry()
        await registry.ensure_fresh()

        def _boom(_slug):
            raise ConnectionError('down')

        monkeypatch.setattr(registry._source, 'read_one', _boom)
        record = await registry.lookup('combo')
        assert record is not None
        return record.status

    assert asyncio.run(_run()) == 'building'


@pytest.mark.parametrize('slug', ['nope'])
def test_unknown_slug_404s_in_both(slug) -> None:
    async def _run() -> tuple[str, str]:
        await seed_default_project()
        set_project_registry(ProjectRegistry())
        return await _guard(slug), await _get_route(slug)

    assert asyncio.run(_run()) == ('project_not_found', 'project_not_found')
