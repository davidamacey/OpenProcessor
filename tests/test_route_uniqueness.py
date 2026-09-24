"""No two routes in the assembled app may share a method + path.

FastAPI serves the first registered match silently, so a duplicate shadows
the later route with no error -- a JSON route once hid the crop image route.
"""

from __future__ import annotations

from collections import Counter

from fastapi.routing import APIRoute


def test_no_two_routes_share_method_and_path() -> None:
    from src.main import app

    pairs = Counter(
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    )
    dupes = sorted(f'{m} {p}' for (m, p), n in pairs.items() if n > 1)
    assert dupes == [], f'shadowed routes: {dupes}'
