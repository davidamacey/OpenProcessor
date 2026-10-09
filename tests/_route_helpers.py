"""Flatten ``app.routes`` for tests.

FastAPI wraps each ``include_router`` in an ``_IncludedRouter`` instead of copying
its routes into ``app.routes``, so a plain ``isinstance(route, APIRoute)`` walk sees
none of the mounted endpoints. ``iter_route_contexts`` yields each endpoint with its
full, prefixed path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.routing import APIRoute, iter_route_contexts


if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.routing import RouteContext


def api_routes(app: FastAPI) -> list[RouteContext]:
    return [
        ctx for ctx in iter_route_contexts(app.routes) if isinstance(ctx.original_route, APIRoute)
    ]
