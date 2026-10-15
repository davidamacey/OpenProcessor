"""Human-write-semantics batch OCC update.

Split out of :mod:`src.clients.occ` to keep that module under the
700-LOC ratchet — a real seam, not an arbitrary split: ``occ.py`` covers
single-doc OCC (:func:`~src.clients.occ.occ_update_one`) and
worker-semantics bulk OCC (:func:`~src.clients.occ.occ_skip_on_conflict_bulk`,
skip-on-conflict); this module covers the third shape, human-write bulk
OCC that retries (rather than skips) a 409.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any, Literal

from src.clients.occ import OCC_BULK_MGET_SOURCE_EXCLUDES, OCC_BULK_PAGE_SIZE
from src.config.curation_accessors import items_index
from src.core.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)


BulkFailure = Literal['conflict', 'transport', 'error']
"""Why a doc was not written: a 409 that outlasted the retries, a bulk request
that failed as a whole, or a non-409 error on the item."""


def _bulk_body(
    index: str, pending: list[tuple[str, dict[str, Any]]], docs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    body: list[dict[str, Any]] = []
    for doc_id, update_doc in pending:
        doc = docs[doc_id]
        body.append(
            {
                'update': {
                    '_index': index,
                    '_id': doc_id,
                    'if_seq_no': doc['_seq_no'],
                    'if_primary_term': doc['_primary_term'],
                }
            }
        )
        body.append({'doc': update_doc})
    return body


def _item_outcome(doc_id: str, item: dict[str, Any]) -> Literal['written', 'conflict', 'error']:
    action = item.get('update') or {}
    status = action.get('status')
    if status in (200, 201):
        return 'written'
    error = action.get('error') or {}
    if status == 409 or 'version_conflict' in error.get('type', ''):
        return 'conflict'
    logger.warning('curation_occ_update_bulk_item_error', doc_id=doc_id, error=str(error or action))
    return 'error'


async def occ_bulk_rounds(
    client: AsyncOpenSearch,
    *,
    index: str,
    ids: list[str],
    prepare: Callable[[str, dict[str, Any] | None], dict[str, Any] | None],
    on_written: Callable[[str], None],
    on_failed: Callable[[str, dict[str, Any] | None, BulkFailure], None],
    max_retries: int,
    refresh: bool | str = False,
    retry_transport: bool = True,
    source_includes: list[str] | None = None,
    source_excludes: list[str] | None = None,
) -> None:
    """The one paged read-merge-conditional-bulk retry loop every bulk OCC
    writer shares; callers only say what to write and what each outcome means.

    Pages ``ids`` at :data:`~src.clients.occ.OCC_BULK_PAGE_SIZE`, one ``_mget`` +
    one conditional ``_bulk`` per page. Ids that hit a 409 are collected across
    pages and retried (re-mget, re-``prepare``) after a jittered backoff, up to
    ``max_retries`` extra rounds. ``refresh`` rides only on the last bulk call
    of each round.

    ``prepare(doc_id, doc)`` gets the mget hit (``None`` when the doc is gone)
    and returns the partial update to write, or ``None`` when it handled the id
    itself (missing, no-op, invalid) and recorded that outcome. ``on_written``
    fires for each applied update. ``on_failed(doc_id, doc, why)`` fires once
    for an id that will not be written: ``doc`` is the last hit read.
    ``retry_transport`` retries the ids of a bulk request that raised.
    """
    from src.clients.curation_opensearch.crops import mget_crops

    pending_ids = list(dict.fromkeys(ids))
    for attempt in range(max_retries + 1):
        if not pending_ids:
            return
        final = attempt == max_retries
        next_round: list[str] = []
        page_starts = list(range(0, len(pending_ids), OCC_BULK_PAGE_SIZE))
        for page_idx, start in enumerate(page_starts):
            page_ids = pending_ids[start : start + OCC_BULK_PAGE_SIZE]
            docs = await mget_crops(
                client,
                page_ids,
                index=index,
                source_includes=source_includes,
                source_excludes=source_excludes,
                seq_no=True,
            )
            pending: list[tuple[str, dict[str, Any]]] = []
            for doc_id in page_ids:
                update_doc = prepare(doc_id, docs.get(doc_id))
                if update_doc is not None:
                    pending.append((doc_id, update_doc))
            if not pending:
                continue

            bulk_body = _bulk_body(index, pending, docs)

            call_refresh = refresh if page_idx == len(page_starts) - 1 else False
            try:
                resp = await client.bulk(body=bulk_body, refresh=call_refresh)
            except Exception as exc:
                logger.warning('curation_occ_update_bulk_transport_error', error=str(exc))
                for doc_id, _update_doc in pending:
                    if retry_transport and not final:
                        next_round.append(doc_id)
                    else:
                        on_failed(doc_id, docs[doc_id], 'transport')
                continue

            for (doc_id, _update_doc), item in zip(pending, resp.get('items') or [], strict=True):
                outcome = _item_outcome(doc_id, item)
                if outcome == 'written':
                    on_written(doc_id)
                elif outcome == 'conflict' and not final:
                    next_round.append(doc_id)
                else:
                    on_failed(doc_id, docs[doc_id], 'error' if outcome == 'error' else 'conflict')

        pending_ids = next_round
        if pending_ids:
            await asyncio.sleep(0.05 + random.random() * 0.15)


async def occ_update_bulk(
    client: AsyncOpenSearch,
    *,
    index: str | None = None,
    ids: list[str],
    merge_fn: Callable[[str, dict[str, Any]], dict[str, Any]],
    max_retries: int = 3,
    refresh: bool | str = False,
) -> dict[str, str]:
    """Human-write-semantics bulk OCC update.

    Unlike :func:`~src.clients.occ.occ_skip_on_conflict_bulk` (worker
    semantics: a 409 means "human wins, skip"), this helper is for batch
    endpoints that are *themselves* a human write (batch label, batch
    move, batch undo, region batch verify) -- a 409 there means a
    concurrent human write raced this one, and both should be retried,
    not silently dropped. The loop is :func:`occ_bulk_rounds`.

    Args:
        client: AsyncOpenSearch instance.
        index: Target index.
        ids: Doc ids to update. Duplicates are de-duplicated, order
            preserved.
        merge_fn: ``(doc_id, source) -> update_doc``. An empty/falsy
            return is a documented noop (recorded as ``'updated'`` with
            nothing written).
        max_retries: Max retry rounds on 409 conflict.
        refresh: OpenSearch refresh policy for the final bulk call of
            each round (``True`` / ``False`` / ``'wait_for'``).

    Returns:
        Per-id status: ``'updated'``, ``'not-found'``, or
        ``'conflict-exhausted'``.
    """
    status: dict[str, str] = {}

    def prepare(doc_id: str, doc: dict[str, Any] | None) -> dict[str, Any] | None:
        if doc is None:
            status[doc_id] = 'not-found'
            return None
        update_doc = merge_fn(doc_id, doc.get('_source') or {})
        if not update_doc:
            status[doc_id] = 'updated'
            return None
        return update_doc

    def on_written(doc_id: str) -> None:
        status[doc_id] = 'updated'

    def on_failed(doc_id: str, _doc: dict[str, Any] | None, _why: BulkFailure) -> None:
        status[doc_id] = 'conflict-exhausted'

    await occ_bulk_rounds(
        client,
        index=index or items_index(),
        ids=ids,
        prepare=prepare,
        on_written=on_written,
        on_failed=on_failed,
        max_retries=max_retries,
        refresh=refresh,
        source_excludes=OCC_BULK_MGET_SOURCE_EXCLUDES,
    )
    return status


__all__ = ['BulkFailure', 'occ_bulk_rounds', 'occ_update_bulk']
