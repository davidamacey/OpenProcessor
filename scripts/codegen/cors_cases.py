#!/usr/bin/env python3
"""Cases for the Rust front's CORS (plan 18a, crate ``op_api``, module ``cors``).

Writes ``contracts/json/cors_cases.json``: a request (method plus the four CORS request headers)
sent through the middleware ``src/core/cors.py`` installs (``add_lan_cors``, Starlette's
``CORSMiddleware``) around a tiny app, and what came back: the status, the body text and every
``access-control-*`` and ``vary`` header. Nothing is typed as an expectation; Starlette is the
definition and ``native/op_api`` is held to it case by case.

The tiny app answers 200 ``app`` to every method on ``/probe`` in two variants, one that sets its
own ``Vary: Accept-Encoding`` (as GZip does in the real app) and one that sets none, because the
middleware appends to the ``Vary`` it finds.

A value with a space around it is not a case: an HTTP server strips it before any application
sees it, so it never reaches the middleware over a socket.

Header values are strings whose characters are the bytes of the value (Latin-1), the way ASGI
carries them. Headers of an answer are a map from lower-case name to the list of its values in
order; the order of different names is not part of a browser's view and is not recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from starlette.requests import Request


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contracts' / 'json' / 'cors_cases.json'

METHODS = ('GET', 'POST', 'OPTIONS')
ALLOWED_ORIGINS = (
    'http://localhost',
    'https://localhost',
    'http://localhost:5184',
    'https://localhost:443',
    'http://127.0.0.1',
    'http://127.0.0.1:4603',
    'http://host.docker.internal',
    'http://host.docker.internal:8080',
    'http://192.168.1.20',
    'https://192.168.0.1:5184',
    'http://10.0.0.5',
    'http://10.255.255.255:9',
    'http://172.16.0.1',
    'http://172.20.3.4:3000',
    'http://172.31.255.255',
)
REFUSED_ORIGINS = (
    'http://172.15.0.1',
    'http://172.32.0.1',
    'https://example.com',
    'http://8.8.8.8',
    'http://localhost/',
    'http://localhost/path',
    'http://localhost:5184/',
    'http://localhost.evil.example',
    'http://localhostx',
    'http://localhost:',
    'http://localhost:80abc',
    'http://192.168.1',
    'http://10.0.0.5.6',
    'http://evil.example/http://localhost',
    'HTTP://localhost',
    'Http://localhost',
    'ftp://localhost',
    'localhost',
    'null',
    'http://localhost:\xe9',
    '',
)
# Preflight origins crossed with every other preflight header; the rest get one request method.
PREFLIGHT_ORIGINS = ('http://localhost:5184', 'http://172.20.3.4:3000', 'https://example.com', '')
REQUEST_METHODS = (
    None,
    'GET',
    'POST',
    'PUT',
    'DELETE',
    'PATCH',
    'HEAD',
    'OPTIONS',
    'QUERY',
    'FOO',
    'get',
    '',
)
REQUEST_HEADERS = (None, 'content-type', 'x-request-id, authorization', '')
PRIVATE_NETWORK = (None, 'true')


def _app(own_vary: bool) -> Any:
    from fastapi import FastAPI
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    from src.core.cors import add_lan_cors

    async def probe(_request: Request) -> PlainTextResponse:
        headers = {'vary': 'Accept-Encoding'} if own_vary else None
        return PlainTextResponse('app', headers=headers)

    app = FastAPI(routes=[Route('/probe', probe, methods=[*METHODS, 'PUT', 'DELETE'])])
    add_lan_cors(app)
    return app


def _call(app: Any, method: str, headers: dict[str, str]) -> dict[str, Any]:
    scope = {
        'type': 'http',
        'asgi': {'version': '3.0'},
        'http_version': '1.1',
        'method': method,
        'scheme': 'http',
        'path': '/probe',
        'raw_path': b'/probe',
        'query_string': b'',
        'root_path': '',
        'headers': [(k.encode('latin-1'), v.encode('latin-1')) for k, v in headers.items()],
        'server': ('testserver', 80),
        'client': ('testclient', 50000),
    }
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {'type': 'http.request', 'body': b'', 'more_body': False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m['type'] == 'http.response.start')
    body = b''.join(m.get('body', b'') for m in sent if m['type'] == 'http.response.body')
    seen: dict[str, list[str]] = {}
    for name, value in start['headers']:
        key = name.decode('latin-1').lower()
        if key.startswith('access-control-') or key == 'vary':
            seen.setdefault(key, []).append(value.decode('latin-1'))
    return {'status': start['status'], 'body': body.decode('latin-1'), 'headers': seen}


def _requests() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in METHODS:
        out.append({'method': method, 'headers': {}})
        out.extend(
            {'method': method, 'headers': {'origin': origin}}
            for origin in (*ALLOWED_ORIGINS, *REFUSED_ORIGINS)
        )
    # a request header that only matters with a preflight is ignored on any other request
    out.append({'method': 'GET', 'headers': {'access-control-request-method': 'POST'}})
    out.append(
        {
            'method': 'GET',
            'headers': {'origin': 'http://localhost', 'access-control-request-method': 'POST'},
        }
    )
    out.append({'method': 'OPTIONS', 'headers': {'access-control-request-method': 'POST'}})
    out.append({'method': 'OPTIONS', 'headers': {'access-control-request-headers': 'x-a'}})
    out.extend(
        {
            'method': 'OPTIONS',
            'headers': {'origin': origin, 'access-control-request-method': 'POST'},
        }
        for origin in (*ALLOWED_ORIGINS, *REFUSED_ORIGINS)
        if origin not in PREFLIGHT_ORIGINS
    )
    for origin in PREFLIGHT_ORIGINS:
        for asked in REQUEST_METHODS:
            for requested in REQUEST_HEADERS:
                for private in PRIVATE_NETWORK:
                    headers: dict[str, str] = {'origin': origin}
                    if asked is not None:
                        headers['access-control-request-method'] = asked
                    if requested is not None:
                        headers['access-control-request-headers'] = requested
                    if private is not None:
                        headers['access-control-request-private-network'] = private
                    out.append({'method': 'OPTIONS', 'headers': headers})
    return out


def render() -> str:
    import starlette

    apps = {True: _app(True), False: _app(False)}
    cases = []
    for own_vary in (False, True):
        for request in _requests():
            answer = _call(apps[own_vary], request['method'], request['headers'])
            cases.append({'app_sets_vary': own_vary, 'request': request, 'answer': answer})
    lines = ',\n'.join(json.dumps(case, sort_keys=True) for case in cases)
    return f'{{"starlette": {json.dumps(starlette.__version__)}, "cases": [\n{lines}\n]}}\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail if the case file is stale')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding='utf-8') != text:
            sys.stderr.write(
                f'{OUT.relative_to(ROOT)} is stale: run scripts/codegen/cors_cases.py\n'
            )
            return 1
        return 0
    OUT.write_text(text, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
