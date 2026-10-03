"""The host the current client used to reach the API, as a ContextVar.

Served links to sibling services (Grafana, MLflow, ...) must point at the
host the BROWSER used (a LAN client opening ``http://10.10.10.20:5184`` needs
``http://10.10.10.20:<service port>``, never ``localhost``). One middleware
captures ``(scheme, host-without-port)`` per request; every consumer reads it
through :func:`current_origin`. ``X-Forwarded-Host`` / ``X-Forwarded-Proto``
(first value) win over ``Host`` / the connection scheme -- they are only
used to build links shown to the same requester.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import TYPE_CHECKING, NamedTuple


if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

_HOST_RE = re.compile(r'^(?:[A-Za-z0-9.\-]+|\[[0-9A-Fa-f:.]+\])$')


class Origin(NamedTuple):
    scheme: str
    host: str


_origin: ContextVar[Origin | None] = ContextVar('request_origin', default=None)


def current_origin() -> Origin | None:
    return _origin.get()


def parse_origin(host_header: str | None, proto_header: str | None, scheme: str) -> Origin | None:
    """``Host`` value (port stripped, validated) + scheme, or ``None``."""
    raw = (host_header or '').split(',')[0].strip()
    host = raw[: raw.index(']') + 1] if raw.startswith('[') and ']' in raw else raw.split(':')[0]
    proto = (proto_header or '').split(',')[0].strip().lower() or scheme
    if not _HOST_RE.match(host) or proto not in ('http', 'https'):
        return None
    return Origin(proto, host)


class RequestOriginMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        headers = {k.decode('latin-1'): v.decode('latin-1') for k, v in scope['headers']}
        origin = parse_origin(
            headers.get('x-forwarded-host') or headers.get('host'),
            headers.get('x-forwarded-proto'),
            scope.get('scheme', 'http'),
        )
        token = _origin.set(origin)
        try:
            await self.app(scope, receive, send)
        finally:
            _origin.reset(token)
