"""The route guard and GET /projects/{slug} read one status function.

The registry revision counter is bumped after the project doc is written, so
a worker whose snapshot still says ``building`` must confirm it at the doc:
at every instant of the flip window the guard and the GET must agree.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from fastapi import HTTPException

from src.services.projects.registry import (
    ProjectRegistry,
    doc_to_record,
    record_to_doc,
    set_project_registry,
)

from .conftest import FakeRegistryOpenSearch, seed_default_project
from .test_registry_pagination import _env, _seed_record  # noqa: F401


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
        client = FakeRegistryOpenSearch()
        await seed_default_project(client)
        _seed_record(client, 'combo')
        client.docs['project:combo'] = record_to_doc(
            replace(doc_to_record(client.docs['project:combo']), status='building')
        )
        client.docs['meta:projects_revision'] = {'revision': 7}
        client.seq['meta:projects_revision'] = 0
        client._refresh_all()
        registry = ProjectRegistry(lambda: client)
        set_project_registry(registry)
        await registry.ensure_fresh()
        seen = [(await _guard('combo'), await _get_route('combo'))]
        assert seen[0] == ('project_building', 'building')
        # The flip: doc is active, revision counter not bumped yet.
        client.docs['project:combo'] = record_to_doc(
            replace(doc_to_record(client.docs['project:combo']), status='active')
        )
        for _ in range(3):
            seen += [(await _guard('combo'), await _get_route('combo'))]
        return seen

    seen = asyncio.run(_run())
    for guard, get in seen[1:]:
        assert (guard, get) == ('active', 'active')


def test_unreachable_doc_keeps_snapshot() -> None:
    async def _run() -> str:
        client = FakeRegistryOpenSearch()
        await seed_default_project(client)
        _seed_record(client, 'combo')
        client.docs['project:combo'] = record_to_doc(
            replace(doc_to_record(client.docs['project:combo']), status='building')
        )
        client.docs['meta:projects_revision'] = {'revision': 7}
        client.seq['meta:projects_revision'] = 0
        client._refresh_all()
        registry = ProjectRegistry(lambda: client)
        await registry.ensure_fresh()

        async def _boom(**_kw):
            raise ConnectionError('down')

        client.get = _boom  # type: ignore[method-assign]
        record = await registry.lookup('combo')
        assert record is not None
        return record.status

    assert asyncio.run(_run()) == 'building'


@pytest.mark.parametrize('slug', ['nope'])
def test_unknown_slug_404s_in_both(slug) -> None:
    async def _run() -> tuple[str, str]:
        client = FakeRegistryOpenSearch()
        await seed_default_project(client)
        set_project_registry(ProjectRegistry(lambda: client))
        return await _guard(slug), await _get_route(slug)

    assert asyncio.run(_run()) == ('project_not_found', 'project_not_found')
