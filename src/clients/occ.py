"""Optimistic concurrency control (OCC) helpers for curation item writes.

The curation pipeline has multiple writers: human PUTs from the
labeler, the region-detection worker cascade, the VLM worker class
fill, the auto_label pipeline, ingest. Without OCC, every writer uses
last-writer-wins semantics, so a worker write 100ms after a human edit
silently clobbers the human's intent.

OCC enforces:
1. Read with ``seq_no_primary_term=True`` to capture the document's
   current version.
2. Update with ``if_seq_no`` + ``if_primary_term`` to assert the version
   has not changed since the read.
3. On ``ConflictError``: re-fetch, re-merge, retry up to ``max_retries``.
4. On final exhaustion: log a structured event and raise
   :class:`OCCFinalConflictError` (human endpoints surface as HTTP 409).

Workers use ``occ_skip_on_conflict_bulk`` instead: on conflict the human
wins, the worker skips the doc silently, and the next poll iteration
sees the updated state. Workers MUST NOT retry — that would silently
overwrite the human's write.

The helpers operate on the configured items index
(``CurationConfig.items_index``) by default but accept any index via
the ``index=`` keyword.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.clients.occ_locks import is_locked_box, is_locked_class, is_locked_item
from src.config import BACKBONE_EMBEDDING_FIELD, ITEM_EMBEDDING_FIELD, get_region_fields
from src.config.curation_accessors import items_index
from src.config.env import env_int
from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)


# Backoff schedule for OCC retries on human endpoints. Workers don't
# retry, so this only applies to ``occ_update_one`` / ``occ_update_bulk``.
_RETRY_BACKOFF_SEC = (0.05, 0.15, 0.45)

# Page size for occ_skip_on_conflict_bulk: doc_ids are chunked into pages
# of this size, and each page costs exactly one `_mget` + one `_bulk`
# round-trip (2 total, vs the prior 2-per-doc GET+update round trips).
# A 347k-doc pass that took 15+ minutes and blew a 600s timeout now
# takes O(N / OCC_BULK_PAGE_SIZE) round trips. Tunable via
# OCC_BULK_PAGE_SIZE env if a caller's payload size or OS cluster needs
# a different page.
OCC_BULK_PAGE_SIZE = env_int('OCC_BULK_PAGE_SIZE', 500)

# Embedding fields are large (512-d+ float vectors) and no existing
# occ_skip_on_conflict_bulk merger reads them (verified: auto_promote,
# the VLM label-batch merger, class-merge writers, and the
# region-detection worker's bulk writer only touch
# class_*/region_*/label_source/cluster_*/history fields) — excluding
# them from the per-page mget keeps the batched fetch small regardless
# of page size.
OCC_BULK_MGET_SOURCE_EXCLUDES = [
    ITEM_EMBEDDING_FIELD,
    BACKBONE_EMBEDDING_FIELD,
    get_region_fields().box_embeddings,
]


class OCCFinalConflictError(Exception):
    """Raised when OCC retries are exhausted on a human-write endpoint.

    The FastAPI handler should catch this and return HTTP 409 with a
    body the labeler can use to surface "another writer edited this
    crop; refresh and retry."
    """

    def __init__(self, doc_id: str, retries: int, last_error: Exception | None = None):
        self.doc_id = doc_id
        self.retries = retries
        self.last_error = last_error
        super().__init__(
            f'OCC conflict on {doc_id} after {retries} retries; last error: {last_error}'
        )


Merger = Callable[[dict[str, Any]], dict[str, Any]]
"""Type alias: takes the current ``_source`` dict, returns a partial
update doc with only the fields to write."""


async def _fetch_with_version(
    client: AsyncOpenSearch,
    *,
    index: str,
    doc_id: str,
) -> tuple[dict[str, Any], int, int]:
    """GET w/ seq_no/primary_term for OCC; excludes embedding vectors
    (no current merge closure reads them). Returns (source, seq_no, primary_term)."""
    resp = await client.get(index=index, id=doc_id, _source_excludes=OCC_BULK_MGET_SOURCE_EXCLUDES)
    source = resp.get('_source') or {}
    seq_no = int(resp.get('_seq_no', 0))
    primary_term = int(resp.get('_primary_term', 0))
    return source, seq_no, primary_term


async def occ_update_one(
    client: AsyncOpenSearch,
    *,
    doc_id: str,
    merger: Merger,
    index: str | None = None,
    max_retries: int = 3,
    refresh: bool | str = False,
    writer_id: str = 'unknown',
) -> dict[str, Any]:
    """OCC-protected single-doc update.

    Args:
        client: AsyncOpenSearch instance.
        doc_id: Document id.
        merger: Callable that receives the current source and returns a
            partial update doc (only the changed fields).
        index: Target index (defaults to the configured items index).
        max_retries: Max retries on ConflictError. Human endpoints use 3.
            Workers use ``occ_skip_on_conflict_bulk`` (max_retries=0).
        refresh: OpenSearch refresh policy. ``True`` for immediate
            visibility (human PUTs), ``False`` for batch writes.
        writer_id: Identifier for structured logs (``human``, ``region_worker``,
            ``vlm_pipeline``, etc.).

    Raises:
        OCCFinalConflictError: when retries are exhausted.

    Returns:
        The final update doc that was successfully written.
    """
    if index is None:
        index = items_index()
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        source, seq_no, primary_term = await _fetch_with_version(client, index=index, doc_id=doc_id)
        update_doc = merger(source)
        try:
            await client.update(
                index=index,
                id=doc_id,
                body={'doc': update_doc},
                if_seq_no=seq_no,
                if_primary_term=primary_term,
                refresh=refresh,
            )
            return update_doc
        except Exception as exc:  # opensearchpy ConflictError or transport error
            last_error = exc
            err_type = type(exc).__name__
            is_conflict = 'Conflict' in err_type or '409' in str(exc)
            if not is_conflict:
                # Non-conflict error — surface immediately, don't retry.
                raise
            logger.info(
                'curation_occ_retry',
                doc_id=doc_id,
                writer_id=writer_id,
                attempt=attempt,
                max_retries=max_retries,
            )
            if attempt < max_retries:
                await asyncio.sleep(_RETRY_BACKOFF_SEC[min(attempt, len(_RETRY_BACKOFF_SEC) - 1)])
                continue
    logger.warning(
        'curation_occ_final_conflict',
        doc_id=doc_id,
        writer_id=writer_id,
        retries=max_retries,
    )
    raise OCCFinalConflictError(doc_id=doc_id, retries=max_retries, last_error=last_error)


async def occ_skip_on_conflict_bulk(
    client: AsyncOpenSearch,
    *,
    doc_ids: list[str],
    merger: Callable[[str, dict[str, Any]], dict[str, Any]],
    index: str | None = None,
    refresh: bool | str = False,
    writer_id: str = 'worker',
    page_size: int | None = None,
) -> dict[str, Any]:
    """Worker-semantics bulk update: skip on conflict (human always wins).

    Batched rewrite (P2-13): the prior implementation paid one GET +
    one conditional UPDATE per doc (2 round trips/doc), which took
    15+ minutes and blew a 600 s timeout on a 347k-doc auto_promote
    pass. This version pages ``doc_ids`` into chunks of ``page_size``
    (default :data:`OCC_BULK_PAGE_SIZE`) and, per page:

    1. One ``_mget`` (via :func:`src.clients.curation_opensearch.crops.mget_crops`)
       fetching every doc's ``_source`` (minus large embedding fields,
       see :data:`OCC_BULK_MGET_SOURCE_EXCLUDES`) + ``_seq_no``/
       ``_primary_term``.
    2. ``merger(doc_id, source)`` computed in-process per fetched doc.
    3. One ``_bulk`` call with a conditional ``update`` action
       (``if_seq_no``/``if_primary_term``) per non-empty merge result.

    That's O(N / page_size) round trips instead of O(N).

    Contract (unchanged from the per-doc implementation):

    - ``merger(doc_id, source) -> update_doc``; ``{}`` counts as
      neither updated nor skipped (a documented noop).
    - Ids missing from the ``_mget`` response are recorded as
      ``{'doc_id':..., 'phase':'fetch', 'error':...}`` in ``errors``.
    - Each bulk response item is classified: 200/201 -> updated;
      409 / ``version_conflict_engine_exception`` -> skipped (the
      human's write stays in place; the worker sees the updated state
      on its next poll) — logs the same structured
      ``curation_worker_skip_human_won`` event, pulling
      ``human_class_validated``/``human_region_validated`` from the
      mget'd source for the audit trail; anything else -> ``errors``
      with ``phase: 'update'``.
    - A ``client.bulk()`` failure (transport/connection error) degrades to
      per-doc ``phase: 'update'`` error entries for that page rather than
      raising — callers already bare-except wrap this function, but a
      raise here would still lose partial progress on other pages.
    - ``refresh`` is forwarded to ``client.bulk(refresh=refresh)``
      unchanged (``True``/``False``/``'wait_for'``).

    Returns:
        ``{updated, updated_ids, skipped_due_to_conflict, skipped_ids, errors}`` --
        ``updated_ids`` names the docs written; ``skipped_ids`` names the docs a conflict skipped, for a caller
        whose write must not be lost (it re-merges and retries them).
    """
    if index is None:
        index = items_index()
    from src.clients.curation_opensearch.crops import mget_crops

    updated_ids: list[str] = []
    skipped = 0
    skipped_ids: list[str] = []
    errors: list[dict[str, Any]] = []

    page = page_size if page_size is not None else OCC_BULK_PAGE_SIZE
    if page <= 0:
        page = len(doc_ids) or 1

    for start in range(0, len(doc_ids), page):
        page_ids = doc_ids[start : start + page]
        if not page_ids:
            continue

        docs = await mget_crops(
            client,
            page_ids,
            index=index,
            source_excludes=OCC_BULK_MGET_SOURCE_EXCLUDES,
            seq_no=True,
        )

        missing = [doc_id for doc_id in page_ids if doc_id not in docs]
        errors.extend(
            {'doc_id': doc_id, 'phase': 'fetch', 'error': 'not_found'} for doc_id in missing
        )

        # doc_id -> (update_doc, source) for every non-empty merge result,
        # in page order so bulk item i maps back to this list's i-th entry.
        pending: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for doc_id in page_ids:
            doc = docs.get(doc_id)
            if doc is None:
                continue
            source = doc.get('_source') or {}
            update_doc = merger(doc_id, source)
            if not update_doc:
                continue
            pending.append((doc_id, update_doc, source))

        if not pending:
            continue

        bulk_body: list[dict[str, Any]] = []
        for doc_id, update_doc, _source in pending:
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

        try:
            resp = await client.bulk(body=bulk_body, refresh=refresh)
        except Exception as exc:
            for doc_id, _update_doc, _source in pending:
                errors.append({'doc_id': doc_id, 'phase': 'update', 'error': str(exc)})
            continue

        items = resp.get('items') or []
        for (doc_id, _update_doc, source), item in zip(pending, items, strict=True):
            action = item.get('update') or {}
            status = action.get('status')
            if status in (200, 201):
                updated_ids.append(doc_id)
                continue
            error = action.get('error') or {}
            err_type = error.get('type', '')
            is_conflict = status == 409 or 'version_conflict' in err_type
            if is_conflict:
                logger.info(
                    'curation_worker_skip_human_won',
                    doc_id=doc_id,
                    writer_id=writer_id,
                    human_class_validated=source.get('class_validated'),
                    human_region_validated=source.get(get_region_fields().validated),
                )
                skipped += 1
                skipped_ids.append(doc_id)
                continue
            errors.append({'doc_id': doc_id, 'phase': 'update', 'error': str(error or action)})

    return {
        'updated': len(updated_ids),
        'updated_ids': updated_ids,
        'skipped_due_to_conflict': skipped,
        'skipped_ids': skipped_ids,
        'errors': errors,
    }


# is_locked_class / _is_locked_marker / is_locked_box / is_locked_item /
# is_human_marker live in occ_locks.py (LOC ratchet) and are imported at
# module top, so every existing `from src.clients.occ import
# is_locked_class`-shaped call site keeps working unchanged.

# Fields an automated class writer must never apply on top of a human-owned
# class row: the one surface every consumer strips/skips (``is_human_owned_class``).
CLASS_WRITE_FIELDS = frozenset(
    {
        'class_id',
        'class_name',
        'class_source',
        'label_source',
        'class_validated',
        'cluster_id',
        'cluster_subid',
        'vlm_confidence',
        'vlm_raw_class',
        'vlm_raw_label',
        'vlm_proposed_class',
        'vlm_class_attempted_at',
        'vlm_class_empty_reason',
        'needs_new_class',
        'class_detector',
        'class_detector_version',
        'class_labeler',
        'class_labeled_at',
        'class_id_history',
    }
)


def strip_class_write_fields(update: dict[str, Any]) -> dict[str, Any]:
    """Drop every ``CLASS_WRITE_FIELDS`` key from an update doc.

    Used by writers that may carry both class and region fields in the
    same update (the region worker's combined path) so a human-label guard
    never leaks into suppressing an unrelated region write — the
    region-detector-scope requirement from Phase 1.
    """
    return {k: v for k, v in update.items() if k not in CLASS_WRITE_FIELDS}


__all__ = [
    'Merger',
    'OCCFinalConflictError',
    'is_locked_box',
    'is_locked_class',
    'is_locked_item',
    'occ_skip_on_conflict_bulk',
    'occ_update_one',
]
