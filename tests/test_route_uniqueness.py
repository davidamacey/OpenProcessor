"""No two routes in the assembled app may share a method + path.

FastAPI serves the first registered match silently, so a duplicate shadows
the later route with no error -- a JSON route once hid the crop image route.
"""

from __future__ import annotations

from collections import Counter

from fastapi.routing import APIRoute


# The global, project-less /health and /events deliberately shadow the
# unscoped `default` alias's scoped routes of the same path (review delta
# 1): the global router is mounted first, and the scoped forms stay
# reachable under /curation/projects/{project}/.
INTENTIONAL_GLOBAL_SHADOWS = frozenset({'GET /curation/health', 'GET /curation/events'})


def test_no_two_routes_share_method_and_path() -> None:
    from src.main import app

    pairs = Counter(
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    )
    dupes = sorted(f'{m} {p}' for (m, p), n in pairs.items() if n > 1)
    assert set(dupes) == INTENTIONAL_GLOBAL_SHADOWS, f'shadowed routes: {dupes}'


def test_intentional_shadows_are_global_first_and_alias_hidden() -> None:
    from src.main import app
    from src.routers.curation.projects import global_router

    global_endpoints = {r.endpoint for r in global_router.routes}
    for shadow in INTENTIONAL_GLOBAL_SHADOWS:
        method, path = shadow.split(' ', 1)
        matches = [
            r
            for r in app.routes
            if isinstance(r, APIRoute) and r.path == path and method in r.methods
        ]
        assert len(matches) == 2, shadow
        first, second = matches
        assert first.endpoint in global_endpoints, f'{shadow}: the global route must win'
        assert first.include_in_schema
        assert not second.include_in_schema, f'{shadow}: the shadowed alias must be hidden'
