"""The one reader for OpenSearch documents that are legitimately absent."""

from __future__ import annotations

from typing import Any

from opensearchpy.exceptions import NotFoundError


async def get_doc_or_none(client: Any, index: str, doc_id: str) -> dict[str, Any] | None:
    """``GET index/_doc/id``, or ``None`` when the document does not exist.

    ``ignore=404`` makes the transport treat the miss as a normal answer, so
    a poll of an optional document never emits a WARNING from the
    ``opensearch`` logger; real failures (5xx, connection errors) still raise
    and are still logged."""
    try:
        resp = await client.get(index=index, id=doc_id, ignore=404)
    except NotFoundError:
        return None
    if not isinstance(resp, dict) or '_source' not in resp:
        return None
    return resp
