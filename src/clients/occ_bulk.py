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
from typing import TYPE_CHECKING, Any

from src.clients.occ import ITEMS_INDEX, OCC_BULK_MGET_SOURCE_EXCLUDES, OCC_BULK_PAGE_SIZE
from src.core.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)


async def occ_update_bulk(
    client: AsyncOpenSearch,
    *,
    index: str = ITEMS_INDEX,
    ids: list[str],
    merge_fn: Callable[[str, dict[str, Any]], dict[str, Any]],
    max_retries: int = 3,
    refresh: bool | str = False,
) -> dict[str, str]:
    """Human-write-semantics bulk OCC update.

    Unlike :func:`~src.clients.occ.occ_skip_on_conflict_bulk` (worker
    semantics: a 409 means "human wins, skip"), this helper is for batch
    endpoints that are *themselves* a human write (batch label, batch
    move, batch undo, region batch verify) — a 409 there means a
    concurrent human write raced this one, and both should be retried,
    not silently dropped.

    Pages ``ids`` at :data:`~src.clients.occ.OCC_BULK_PAGE_SIZE`, one
    ``_mget`` + one conditional ``_bulk`` per page. Ids that hit a 409
    conflict are collected across all pages and retried (re-mget +
    re-merge) with jittered backoff, up to ``max_retries`` rounds.

    ``refresh`` is only ever attached to the last bulk call issued in a
    given round (not per page, and never per item) — callers doing a
    single page (the common case) get exactly one refresh-carrying bulk
    call.

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
    from src.clients.curation_opensearch import mget_crops

    status: dict[str, str] = {}
    pending_ids = list(dict.fromkeys(ids))
    page = OCC_BULK_PAGE_SIZE

    for attempt in range(max_retries + 1):
        if not pending_ids:
            break
        next_round: list[str] = []
        page_starts = list(range(0, len(pending_ids), page))

        for page_idx, start in enumerate(page_starts):
            page_ids = pending_ids[start : start + page]
            docs = await mget_crops(
                client,
                page_ids,
                index=index,
                source_excludes=OCC_BULK_MGET_SOURCE_EXCLUDES,
                seq_no=True,
            )

            for doc_id in page_ids:
                if doc_id not in docs:
                    status[doc_id] = 'not-found'

            pending: list[tuple[str, dict[str, Any]]] = []
            for doc_id in page_ids:
                doc = docs.get(doc_id)
                if doc is None:
                    continue
                source = doc.get('_source') or {}
                update_doc = merge_fn(doc_id, source)
                if not update_doc:
                    status[doc_id] = 'updated'
                    continue
                pending.append((doc_id, update_doc))

            if not pending:
                continue

            bulk_body: list[dict[str, Any]] = []
            for doc_id, update_doc in pending:
                doc = docs[doc_id]
                bulk_body.append(
                    {
                        'update': {
                            '_index': index,
                            '_id': doc_id,
                            'if_seq_no': doc['_seq_no'],
                            'if_primary_term': doc['_primary_term'],
                        }
                    }
                )
                bulk_body.append({'doc': update_doc})

            is_last_page_of_round = page_idx == len(page_starts) - 1
            call_refresh = refresh if is_last_page_of_round else False
            try:
                resp = await client.bulk(body=bulk_body, refresh=call_refresh)
            except Exception as exc:
                logger.warning('curation_occ_update_bulk_transport_error', error=str(exc))
                for doc_id, _update_doc in pending:
                    next_round.append(doc_id)
                continue

            items = resp.get('items') or []
            for (doc_id, _update_doc), item in zip(pending, items, strict=True):
                action = item.get('update') or {}
                item_status = action.get('status')
                if item_status in (200, 201):
                    status[doc_id] = 'updated'
                    continue
                error = action.get('error') or {}
                err_type = error.get('type', '')
                is_conflict = item_status == 409 or 'version_conflict' in err_type
                if is_conflict:
                    next_round.append(doc_id)
                else:
                    logger.warning(
                        'curation_occ_update_bulk_item_error',
                        doc_id=doc_id,
                        error=str(error or action),
                    )
                    status[doc_id] = 'conflict-exhausted'

        pending_ids = next_round
        if pending_ids and attempt < max_retries:
            backoff = 0.05 + random.random() * 0.15
            await asyncio.sleep(backoff)

    for doc_id in pending_ids:
        status[doc_id] = 'conflict-exhausted'

    return status


__all__ = ['occ_update_bulk']
