"""Scroll-context cleanup.

OpenSearch keeps a scroll's search context open until ``scroll`` expires; a loop that raises
midway must still release it ("close the search context when you're done"):
https://docs.opensearch.org/latest/api-reference/search-apis/scroll/
"""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy.exceptions import OpenSearchException


logger = logging.getLogger(__name__)


async def clear_scroll_quietly(client: Any, scroll_id: str | None) -> None:
    """Release ``scroll_id`` if one was opened. A failure here is logged, never raised: it must
    not replace the error that sent the caller into its ``finally``."""
    if not scroll_id:
        return
    try:
        await client.clear_scroll(scroll_id=scroll_id)
    except OpenSearchException as exc:
        logger.debug(f'clear_scroll failed: {exc}')


async def scroll_ids(
    client: Any, *, index: str, query: dict[str, Any], size: int = 500, keep_alive: str = '2m'
) -> list[str]:
    """Every ``_id`` matching ``query``, with the scroll context cleared on success and error."""
    ids: list[str] = []
    scroll_id: str | None = None
    try:
        resp = await client.search(
            index=index, body={'size': size, 'query': query, '_source': False}, scroll=keep_alive
        )
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']
        while hits:
            ids.extend(h['_id'] for h in hits)
            resp = await client.scroll(scroll_id=scroll_id, scroll=keep_alive)
            scroll_id = resp.get('_scroll_id')
            hits = resp['hits']['hits']
    finally:
        await clear_scroll_quietly(client, scroll_id)
    return ids
