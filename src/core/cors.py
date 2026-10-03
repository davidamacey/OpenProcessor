"""CORS for the API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.middleware.cors import CORSMiddleware


if TYPE_CHECKING:
    from fastapi import FastAPI


def add_lan_cors(application: FastAPI) -> None:
    """Allow a labeler/curation frontend and any LAN client to reach the API.

    In production a reverse proxy usually handles routing so cross-origin
    calls are rare, but this covers dev mode (vite/webpack dev servers on a
    different port), direct API access from LAN IPs, and other internal
    network clients. Without it a cross-origin frontend fails with "Failed
    to fetch"/no CORS headers even though the server logs a 200.
    """
    application.add_middleware(
        CORSMiddleware,
        allow_origin_regex=(
            r'^https?://(localhost|127\.0\.0\.1|host\.docker\.internal'
            r'|192\.168\.\d+\.\d+'  # RFC-1918 class C
            r'|10\.\d+\.\d+\.\d+'  # RFC-1918 class A
            r'|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+'  # RFC-1918 class B
            r')(:\d+)?$'
        ),
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
        expose_headers=['X-Request-ID', 'X-Process-Time'],
    )
