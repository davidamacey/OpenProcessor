"""``reset_clusters`` of the cluster bench survives a long reset (#357).

A whole-index ``update_by_query`` held open for the length of the reset failed after about
34 minutes at 250,000 rows with ``ConnectionError(400, 'Too many headers received')``: the
one response of a request that long carries more headers than the client accepts (aiohttp
stops at 128; the node adds a ``Warning`` header per deprecation it logged meanwhile). The
node is not needed to show the client side of that, so the server below sends the headers.
"""

from __future__ import annotations

from typing import Any

import pytest
from aiohttp import web
from opensearchpy import AIOHttpConnection, AsyncOpenSearch
from opensearchpy.exceptions import ConnectionError as OsConnectionError

from scripts.bench import cluster_plumbing_bench as bench


pytestmark = pytest.mark.asyncio

WARNING_HEADERS = [('Warning', f'299 OpenSearch "deprecation {i}"') for i in range(200)]


class _Node:
    """An OpenSearch stand-in. A synchronous update_by_query answers with the long run's
    header flood; an asynchronous one returns a task id, and the task endpoint reports it."""

    def __init__(self, *, remaining: list[int], task_polls: int = 2, task_body: Any = None) -> None:
        self.remaining = remaining  # documents still holding a cluster field, per count call
        self.task_polls = task_polls
        self.task_body = task_body
        self.update_params: list[dict[str, str]] = []
        self.update_bodies: list[Any] = []

    async def update_by_query(self, request: web.Request) -> web.Response:
        self.update_params.append(dict(request.query))
        self.update_bodies.append(await request.json())
        if request.query.get('wait_for_completion') == 'false':
            return web.json_response({'task': 'node:7'})
        return web.json_response({'updated': 1}, headers=WARNING_HEADERS)

    async def task(self, request: web.Request) -> web.Response:
        assert request.match_info['task'] == 'node:7'
        self.task_polls -= 1
        done = self.task_polls <= 0
        body = {'completed': done, 'task': {'id': 7}}
        if done:
            body['response'] = self.task_body or {
                'updated': 10,
                'version_conflicts': 0,
                'failures': [],
            }
        return web.json_response(body)

    async def count(self, request: web.Request) -> web.Response:  # noqa: ARG002
        return web.json_response({'count': self.remaining.pop(0) if self.remaining else 0})

    async def refresh(self, request: web.Request) -> web.Response:  # noqa: ARG002
        return web.json_response({})


async def _serve(node: _Node) -> tuple[web.AppRunner, AsyncOpenSearch]:
    app = web.Application()
    app.router.add_post('/{index}/_update_by_query', node.update_by_query)
    app.router.add_get('/_tasks/{task}', node.task)
    app.router.add_post('/{index}/_count', node.count)
    app.router.add_get('/{index}/_count', node.count)
    app.router.add_post('/{index}/_refresh', node.refresh)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]  # ephemeral port
    client = AsyncOpenSearch(hosts=[f'http://127.0.0.1:{port}'], connection_class=AIOHttpConnection)
    return runner, client


async def test_a_held_open_update_by_query_dies_on_the_header_flood() -> None:
    """The failure of the issue, reproduced: this is what the old reset did."""
    runner, client = await _serve(_Node(remaining=[]))
    try:
        with pytest.raises(OsConnectionError, match='Too many headers received'):
            await client.update_by_query(index='items', body={}, wait_for_completion=True)
    finally:
        await client.close()
        await runner.cleanup()


async def test_reset_clusters_uses_a_task_and_survives_the_flood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bench, 'RESET_POLL_S', 0.0)
    node = _Node(remaining=[0])
    runner, client = await _serve(node)
    try:
        await bench.reset_clusters(client)
    finally:
        await client.close()
        await runner.cleanup()
    assert node.update_params[0]['wait_for_completion'] == 'false'
    assert node.update_params[0]['slices'] == 'auto'
    # Only documents that hold a cluster field are rewritten.
    assert 'match_all' not in str(node.update_bodies[0]['query'])


async def test_reset_clusters_fails_when_the_task_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bench, 'RESET_POLL_S', 0.0)
    body = {'updated': 3, 'version_conflicts': 0, 'failures': [{'cause': {'type': 'boom'}}]}
    runner, client = await _serve(_Node(remaining=[0], task_body=body))
    try:
        with pytest.raises(bench.ResetError, match='failures'):
            await bench.reset_clusters(client)
    finally:
        await client.close()
        await runner.cleanup()


async def test_reset_clusters_fails_when_documents_still_hold_cluster_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """conflicts=proceed skips a document a writer changed meanwhile: the reset must not
    report done while it is still clustered."""
    monkeypatch.setattr(bench, 'RESET_POLL_S', 0.0)
    runner, client = await _serve(_Node(remaining=[12]))
    try:
        with pytest.raises(bench.ResetError, match='12'):
            await bench.reset_clusters(client)
    finally:
        await client.close()
        await runner.cleanup()
