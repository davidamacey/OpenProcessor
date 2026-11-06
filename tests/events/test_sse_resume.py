"""SSE frames carry id, event and data; a reconnect with Last-Event-ID replays exactly what was missed."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import httpx
import psycopg
import pytest
import uvicorn
from _events_support import seed_project
from _listener_support import running_listener
from fastapi import FastAPI, Header
from fastapi.sse import EventSourceResponse

from src.routers.curation._event_streams import to_sse
from src.services.events.cursor import Cursor
from src.services.events.publish import apublish
from src.services.events.sse import event_stream


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.services.events.listener import EventListener


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


def _app(listener: EventListener) -> FastAPI:
    app = FastAPI()

    @app.get('/events/{slug}', response_class=EventSourceResponse)
    async def events(slug: str, last_event_id: str | None = Header(default=None)) -> Any:
        sub = await listener.subscribe(slug)
        try:
            async for frame in event_stream(sub, last_event_id):
                yield to_sse(frame)
        finally:
            await listener.unsubscribe(sub)

    return app


@asynccontextmanager
async def _serve(listener: EventListener) -> AsyncIterator[str]:
    server = uvicorn.Server(uvicorn.Config(_app(listener), port=0, log_level='warning'))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f'http://127.0.0.1:{port}'
    finally:
        server.should_exit = True
        await task


async def _subscribed(listener: EventListener, slug: str, count: int = 1) -> None:
    async with asyncio.timeout(30):
        while listener.stats(slug)['subscribers'] < count:
            await asyncio.sleep(0.05)


async def _frames(url: str, last_event_id: str | None, count: int) -> list[dict[str, Any]]:
    """Read ``count`` frames off one connection, then close it."""
    headers = {'Last-Event-ID': last_event_id} if last_event_id else {}
    frames: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    async with (
        httpx.AsyncClient(timeout=40) as client,
        client.stream('GET', url, headers=headers) as response,
    ):
        assert response.headers['content-type'].startswith('text/event-stream')
        async for line in response.aiter_lines():
            if line.startswith(':'):
                continue
            if line == '':
                if current:
                    frames.append(current)
                    current = {}
                    if len(frames) == count:
                        return frames
                continue
            key, _, value = line.partition(': ')
            current[key] = json.loads(value) if key == 'data' else value
    return frames


@pytest.mark.asyncio
async def test_resume_replays_41_to_100_in_order_without_duplicates(pg_env, broker) -> None:
    seed_project(pg_env, 'resume')
    async with running_listener(broker, tick_s=0.3) as listener, _serve(listener) as base:
        url = f'{base}/events/resume'
        first = asyncio.create_task(_frames(url, None, 40))
        await _subscribed(listener, 'resume')
        for i in range(1, 101):
            await apublish('crop.created', {'crop_id': str(i)}, project='resume')
        seen = await asyncio.wait_for(first, 40)
        assert [f['data']['crop_id'] for f in seen] == [str(i) for i in range(1, 41)]
        assert all(f['event'] == 'crop.created' and '-' in f['id'] for f in seen)
        assert seen[0]['data']['type'] == 'crop.created'
        assert seen[0]['data']['project'] == 'resume'

        await asyncio.sleep(0.5)  # every row of the burst is committed and visible
        resumed = await _frames(url, seen[-1]['id'], 60)
        assert [f['data']['crop_id'] for f in resumed] == [str(i) for i in range(41, 101)]
        ids = [Cursor.parse(f['id']) for f in resumed]
        assert ids == sorted(set(ids))


@pytest.mark.asyncio
async def test_a_resumed_stream_continues_live_after_the_replay(pg_env, broker) -> None:
    seed_project(pg_env, 'live-after')
    async with running_listener(broker, tick_s=0.3) as listener, _serve(listener) as base:
        url = f'{base}/events/live-after'
        head = asyncio.create_task(_frames(url, None, 1))
        await _subscribed(listener, 'live-after')
        await apublish('crop.created', {'crop_id': '1'}, project='live-after')
        first = (await asyncio.wait_for(head, 40))[0]
        await apublish('crop.created', {'crop_id': '2'}, project='live-after')
        await asyncio.sleep(0.5)
        resumed = asyncio.create_task(_frames(url, first['id'], 2))
        await _subscribed(listener, 'live-after')
        await apublish('crop.created', {'crop_id': '3'}, project='live-after')
        frames = await asyncio.wait_for(resumed, 40)
    assert [f['data']['crop_id'] for f in frames] == ['2', '3']


@pytest.mark.asyncio
async def test_a_cursor_older_than_the_oldest_retained_row_gets_one_reset_first(
    pg_env, broker
) -> None:
    project = seed_project(pg_env, 'expired')
    async with running_listener(broker, tick_s=0.3) as listener, _serve(listener) as base:
        for i in range(3):
            await apublish('crop.created', {'crop_id': str(i)}, project='expired')
        async with asyncio.timeout(30):  # the xmin horizon is cluster-wide: wait for visibility
            while len(rows := await listener.read_after(project, Cursor(0, 0))) < 3:
                await asyncio.sleep(0.2)
        old = rows[0]
        with psycopg.connect(pg_env.conninfo('super'), autocommit=True) as conn:
            conn.execute('DELETE FROM events WHERE project_id = %s AND id <= %s', (project, old.id))
        frames = await _frames(f'{base}/events/expired', Cursor(old.xid, old.id).format(), 1)
    assert frames[0]['event'] == 'reset'
    assert frames[0]['data'] == {'reason': 'cursor_expired'}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('last_event_id', 'reason'),
    [('garbage', 'cursor_invalid'), ('99999999999999-1', 'cursor_unknown')],
)
async def test_an_invalid_or_foreign_cursor_gets_a_reset(
    pg_env, broker, last_event_id: str, reason: str
) -> None:
    seed_project(pg_env, f'reset-{reason[7:10]}')
    async with running_listener(broker, tick_s=0.3) as listener, _serve(listener) as base:
        frames = await _frames(f'{base}/events/reset-{reason[7:10]}', last_event_id, 1)
    assert frames[0]['event'] == 'reset'
    assert frames[0]['data'] == {'reason': reason}
