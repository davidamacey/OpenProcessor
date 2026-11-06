"""A stream holds no pooled connection between reads, and no streaming route takes DbSessionDep."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from src.db.engine import get_async_engine
from src.db.session import async_session_factory


REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMING_NAMES = {'StreamingResponse', 'EventSourceResponse'}


def streaming_routes_using_the_request_session(source: str) -> list[str]:
    """Names of functions that take ``DbSessionDep`` and also build a streaming response."""
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        takes_session = any(
            'DbSessionDep' in ast.unparse(arg.annotation)
            for arg in [*node.args.args, *node.args.kwonlyargs]
            if arg.annotation is not None
        )
        streams = any(
            isinstance(n, ast.Name) and n.id in STREAMING_NAMES for n in ast.walk(node)
        ) or any(
            isinstance(n, ast.Constant) and n.value == 'text/event-stream' for n in ast.walk(node)
        )
        if takes_session and streams:
            offenders.append(node.name)
    return offenders


def test_the_guard_flags_a_streaming_route_that_takes_the_request_session() -> None:
    planted = (
        'from fastapi.responses import StreamingResponse\n'
        'async def feed(session: DbSessionDep):\n'
        '    return StreamingResponse(gen())\n'
    )
    assert streaming_routes_using_the_request_session(planted) == ['feed']
    fine = 'async def rows(session: DbSessionDep):\n    return {}\n'
    assert streaming_routes_using_the_request_session(fine) == []


def test_no_route_in_the_tree_streams_with_the_request_session() -> None:
    offenders = {
        str(path.relative_to(REPO_ROOT)): found
        for path in (REPO_ROOT / 'src').rglob('*.py')
        if (found := streaming_routes_using_the_request_session(path.read_text(encoding='utf-8')))
    }
    assert offenders == {}


@pytest.mark.db
@pytest.mark.asyncio
async def test_an_open_stream_holds_no_pooled_connection_between_reads(
    pg_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_DB_API_CONNECTIONS_TOTAL', '4')
    monkeypatch.setenv('WEB_CONCURRENCY', '4')  # a pool of one connection
    factory = async_session_factory()
    between_reads: list[int] = []

    async def events():
        for _ in range(2):
            async with factory() as session:  # one session per read, never per stream
                value = (await session.execute(text('SELECT 1'))).scalar_one()
            between_reads.append(get_async_engine().sync_engine.pool.checkedout())
            await asyncio.sleep(0.05)
            yield f'data: {value}\n\n'

    app = FastAPI()

    @app.get('/stream')
    async def stream() -> StreamingResponse:
        return StreamingResponse(events(), media_type='text/event-stream')

    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url='http://t') as client,
        client.stream('GET', '/stream') as response,
    ):
        chunks = [chunk async for chunk in response.aiter_text()]
    assert between_reads == [0, 0]
    assert len(chunks) >= 1
