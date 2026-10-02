"""Found live: ``GET /clusters/balance/{index}`` answered 422 because the
generic ``GET /clusters/{index}/{cluster_id}`` was registered first and took
the request (``index='balance'``, ``cluster_id='global'``)."""

from __future__ import annotations

from fastapi.routing import APIRoute
from starlette.routing import Match

from src.routers.clusters import router


def _first_match(path: str, method: str = 'GET') -> str:
    scope = {'type': 'http', 'path': path, 'method': method}
    for route in router.routes:
        if isinstance(route, APIRoute):
            match, _ = route.matches(scope)
            if match == Match.FULL:
                return route.path
    raise AssertionError(f'no route for {method} {path}')


def test_literal_two_segment_routes_win_over_the_member_listing() -> None:
    assert _first_match('/clusters/balance/global') == '/clusters/balance/{index}'
    assert _first_match('/clusters/stats/faces') == '/clusters/stats/{index}'


def test_the_member_listing_still_serves_its_own_paths() -> None:
    assert _first_match('/clusters/global/3') == '/clusters/{index}/{cluster_id}'
