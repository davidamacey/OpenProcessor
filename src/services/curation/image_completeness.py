"""The one place that checks whether a stored photo is whole, and the edits of its ``item_ids``.

The rule is :mod:`src.domain.image_completeness`: the images doc lists its items, and a photo is
whole when every listed item exists. This module reads the items (:func:`missing_items`, used
by the content lookup and by the writer's 409 answer, so one rule decides a ``duplicate``) and
makes the edits of the list, each an OCC write on the images doc through
:func:`~src.clients.occ_bulk.occ_update_bulk` (read, merge, conditional write with
``if_seq_no`` / ``if_primary_term``, a 409 re-read):

* :func:`set_item_ids` is the repair's last write, which also clears the failure mark;
* :func:`mark_incomplete` is the writer's best-effort note on a photo it saw fail.

OpenSearch documents used: multi-get (``_source=false`` and ``_source_includes``; a get is real
time, so an item written but not refreshed is found:
https://docs.opensearch.org/latest/api-reference/document-apis/multi-get/ and
https://docs.opensearch.org/latest/api-reference/document-apis/get-documents/), the bulk API
(``if_seq_no`` / ``if_primary_term`` on an action:
https://docs.opensearch.org/latest/api-reference/document-apis/bulk/) and the update API
(a partial ``doc`` is merged into the stored one:
https://docs.opensearch.org/latest/api-reference/document-apis/update-document/).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException
from sqlalchemy.exc import SQLAlchemyError

from src.clients.occ_bulk import occ_update_bulk
from src.core.logging import get_logger
from src.domain.image_completeness import INCOMPLETE_MARK_FIELD, ITEM_IDS_FIELD
from src.services.curation.ops_metrics import count_whole_event
from src.services.embedding_spaces.readers import vector_source_excludes


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


logger = get_logger(__name__)

MGET_PAGE = 1000
# 409 retry rounds of the two writes a writer makes while it is the live writer of a photo; the
# repair age (``ingest_writer.REPAIR_AFTER_S``) counts their requests.
LIST_WRITE_RETRIES = 2
MARK_WRITE_RETRIES = 1


async def missing_items(
    client: Any, items_index: str, listed: Mapping[str, Sequence[str]]
) -> dict[str, list[str]]:
    """``{image_id: its listed items that do not exist}`` for the images that miss any.

    One real-time ``mget`` per page of :data:`MGET_PAGE` ids over the items of every image of
    ``listed``, with the source switched off."""
    wanted = sorted({item for ids in listed.values() for item in ids})
    found: set[str] = set()
    for start in range(0, len(wanted), MGET_PAGE):
        resp = await client.mget(
            index=items_index, body={'ids': wanted[start : start + MGET_PAGE]}, _source=False
        )
        found.update(d['_id'] for d in resp.get('docs') or [] if d.get('found'))
    gaps = {image_id: [i for i in ids if i not in found] for image_id, ids in listed.items()}
    return {image_id: gap for image_id, gap in gaps.items() if gap}


async def set_item_ids(client: Any, images_index: str, image_id: str, item_ids: list[str]) -> None:
    """Repair's last write: the list becomes the items the repair wrote, and the failure mark is
    cleared in the same write. Raises when the doc is gone or the write did not land."""

    def merge(_id: str, source: dict[str, Any]) -> dict[str, Any]:
        update: dict[str, Any] = {ITEM_IDS_FIELD: sorted(item_ids)}
        if source.get(INCOMPLETE_MARK_FIELD):
            update[INCOMPLETE_MARK_FIELD] = False
        return update

    status = await occ_update_bulk(
        client,
        index=images_index,
        ids=[image_id],
        merge_fn=merge,
        vector_excludes=await vector_source_excludes('images'),
        max_retries=LIST_WRITE_RETRIES,
    )
    if status.get(image_id) != 'updated':
        msg = f'item_ids of image {image_id} not set: {status.get(image_id)}'
        raise OpenSearchException(msg)


async def mark_incomplete(client: Any, images_index: str, image_id: str) -> None:
    """Note on the images doc that its own writer saw the items step fail, so the next ingest
    of the same bytes repairs it at once instead of waiting out the repair age. Best effort: a
    write that fails is logged and counted, never raised (the age rule covers it)."""
    try:
        status = await occ_update_bulk(
            client,
            index=images_index,
            ids=[image_id],
            merge_fn=lambda _id, source: (
                {} if source.get(INCOMPLETE_MARK_FIELD) else {INCOMPLETE_MARK_FIELD: True}
            ),
            vector_excludes=await vector_source_excludes('images'),
            max_retries=MARK_WRITE_RETRIES,
        )
    except (OpenSearchException, SQLAlchemyError) as exc:
        logger.warning('ingest_incomplete_mark_failed', image_id=image_id, error=str(exc))
        count_whole_event('mark_failed')
        return
    if status.get(image_id) == 'updated':
        count_whole_event('marked')
    else:
        logger.warning('ingest_incomplete_mark_failed', image_id=image_id, status=status)
        count_whole_event('mark_failed')


__all__ = ['mark_incomplete', 'missing_items', 'set_item_ids']
