"""GET /projects is bounded: a default page size, a served maximum, and the limit refused above it.
Offline: the registry and the page read are faked."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError

from src.routers.curation import projects as projects_router
from src.services.projects import listing


class _Registry:
    async def ensure_fresh(self) -> None:
        return None

    def retired_slugs(self) -> frozenset[str]:
        return frozenset()


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
    limits: list[int | None] = []

    async def fake_fetch(_session: Any, *, limit: int | None, **_: Any) -> tuple[list, None, int]:
        limits.append(limit)
        return [], None, 0

    async def no_opensearch() -> Any:
        raise OpenSearchConnectionError('N/A', 'offline', OSError('none'))

    monkeypatch.setattr(projects_router, 'get_project_registry', lambda: _Registry())
    monkeypatch.setattr(projects_router.projects_listing, 'fetch_page', fake_fetch)
    monkeypatch.setattr(projects_router, 'make_curation_opensearch', no_opensearch)
    return limits


def test_the_bounds_are_served(seen: list[int | None]) -> None:
    res = asyncio.run(projects_router.list_projects(session=None))  # type: ignore[arg-type]
    assert (res.default_limit, res.max_limit) == (listing.DEFAULT_LIMIT, listing.MAX_LIMIT)
    assert 1 <= res.default_limit <= res.max_limit


def test_the_route_refuses_a_limit_above_the_served_maximum() -> None:
    app = FastAPI()
    app.include_router(projects_router.global_router)
    params = app.openapi()['paths']['/projects']['get']['parameters']
    limit = next(p for p in params if p['name'] == 'limit')
    assert limit['schema']['anyOf'][0]['maximum'] == listing.MAX_LIMIT
