"""Pages of search hits read by several point-in-time slices at once.

One scroll is one cursor: each page waits for the one before. ``n_slices`` slices of a
point-in-time view are independent cursors over disjoint documents, so the node renders
pages for several of them at the same time while the caller works on the page it has.
Pages arrive in no fixed order (hits carry their ``sort`` key when the caller needs one);
memory is bounded by ``n_slices`` pages in the queue plus the one being handled.

Where a point in time cannot be opened (an older node, a test double) the pages come
from a plain scroll instead, in index order.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.core.logging import get_logger
from src.utils.task_group import run_all


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)

KEEP_ALIVE = '5m'
DEFAULT_SLICES = 8
DEFAULT_PAGE_SIZE = 2000


def _fields_body(source: Sequence[str], docvalue_fields: Sequence[str]) -> dict[str, Any]:
    """The ``_source`` / ``docvalue_fields`` part of a search body (no ``_source`` when none is wanted)."""
    body: dict[str, Any] = {'_source': list(source) if source else False}
    if docvalue_fields:
        body['docvalue_fields'] = list(docvalue_fields)
    return body


async def _scroll_pages(
    client: AsyncOpenSearch,
    index: str,
    query: dict[str, Any],
    fields: dict[str, Any],
    page_size: int,
) -> AsyncGenerator[list[dict[str, Any]], None]:
    body: dict[str, Any] = {'size': page_size, 'query': query, **fields}
    resp = await client.search(index=index, body=body, scroll='2m')
    scroll_id = resp.get('_scroll_id')
    try:
        while True:
            hits = resp['hits']['hits']
            if not hits:
                break
            yield hits
            if not scroll_id:
                break
            resp = await client.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = resp.get('_scroll_id')
    finally:
        if scroll_id:
            try:
                await client.clear_scroll(scroll_id=scroll_id)
            except OpenSearchException as exc:
                logger.debug('curation_pit_pages_clear_scroll_failed', error=str(exc))


async def iter_pages(
    client: AsyncOpenSearch,
    *,
    index: str,
    query: dict[str, Any],
    source: Sequence[str],
    docvalue_fields: Sequence[str] = (),
    n_slices: int = DEFAULT_SLICES,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> AsyncGenerator[list[dict[str, Any]], None]:
    """Yield each page of hits of ``query``: ``_source`` limited to ``source``, and the
    ``docvalue_fields`` (a ``knn_vector`` field: base64 float32 under ``fields``, OpenSearch 3.7,
    https://opensearch.org/blog/retrieve-vectors-5x-faster-with-docvalue_fields-in-opensearch/)."""
    fields = _fields_body(source, docvalue_fields)
    try:
        pit_resp = await client.create_pit(index=index, keep_alive=KEEP_ALIVE)
        pit_id = pit_resp.get('pit_id') or pit_resp.get('pit')
        if not pit_id:
            raise RuntimeError(f'create_pit returned no pit_id: {pit_resp!r}')
    except (OpenSearchException, RuntimeError) as exc:
        logger.debug('curation_pit_pages_unavailable_scroll', error=str(exc))
        async for page in _scroll_pages(client, index, query, fields, page_size):
            yield page
        return

    queue: asyncio.Queue[list[dict[str, Any]] | None] = asyncio.Queue(maxsize=n_slices)

    async def produce(slice_id: int) -> None:
        search_after: Any = None
        while True:
            body: dict[str, Any] = {
                'size': page_size,
                'query': query,
                **fields,
                'pit': {'id': pit_id, 'keep_alive': KEEP_ALIVE},
                'sort': [{'_shard_doc': 'asc'}],
            }
            if n_slices > 1:
                body['slice'] = {'id': slice_id, 'max': n_slices}
            if search_after is not None:
                body['search_after'] = search_after
            resp = await client.search(body=body)
            hits = (resp.get('hits') or {}).get('hits') or []
            if not hits:
                return
            await queue.put(hits)
            search_after = hits[-1].get('sort')
            if not search_after:
                return

    async def run() -> None:
        try:
            await run_all([produce(i) for i in range(n_slices)])
        except Exception:
            # wake the consumer, which re-raises through ``await task``; a cancel (the consumer
            # left) must not wait on a full queue, so it is not caught here
            await queue.put(None)
            raise
        await queue.put(None)

    task = asyncio.create_task(run())
    try:
        while (got := await queue.get()) is not None:
            yield got
        await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        try:
            await client.delete_pit(body={'pit_id': [pit_id]})
        except OpenSearchException as exc:
            logger.debug('curation_pit_pages_delete_failed', error=str(exc))
