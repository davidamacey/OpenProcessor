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
import os
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config, get_region_fields
from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

ITEMS_INDEX = get_curation_config().items_index

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
OCC_BULK_PAGE_SIZE = int(os.environ.get('OCC_BULK_PAGE_SIZE', '500'))

# Embedding fields are large (512-d+ float vectors) and no existing
# occ_skip_on_conflict_bulk merger reads them (verified: auto_promote,
# the VLM label-batch merger, class-merge writers, and the
# region-detection worker's bulk writer only touch
# class_*/region_*/label_source/cluster_*/history fields) — excluding
# them from the per-page mget keeps the batched fetch small regardless
# of page size.
OCC_BULK_MGET_SOURCE_EXCLUDES = ['pe_embedding', 'v6_embedding', get_region_fields().embedding]


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
    """GET with seq_no/primary_term for OCC. Returns (source, seq_no, primary_term)."""
    resp = await client.get(index=index, id=doc_id)
    source = resp.get('_source') or {}
    seq_no = int(resp.get('_seq_no', 0))
    primary_term = int(resp.get('_primary_term', 0))
    return source, seq_no, primary_term


async def occ_update_one(
    client: AsyncOpenSearch,
    *,
    doc_id: str,
    merger: Merger,
    index: str = ITEMS_INDEX,
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
        writer_id: Identifier for structured logs (``human``, ``sam_worker``,
            ``vlm_pipeline``, etc.).

    Raises:
        OCCFinalConflictError: when retries are exhausted.

    Returns:
        The final update doc that was successfully written.
    """
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
                'legacy_occ_retry',
                doc_id=doc_id,
                writer_id=writer_id,
                attempt=attempt,
                max_retries=max_retries,
            )
            if attempt < max_retries:
                await asyncio.sleep(_RETRY_BACKOFF_SEC[min(attempt, len(_RETRY_BACKOFF_SEC) - 1)])
                continue
    logger.warning(
        'legacy_occ_final_conflict',
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
    index: str = ITEMS_INDEX,
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

    1. One ``_mget`` (via :func:`src.clients.curation_opensearch.mget_crops`)
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
      ``legacy_worker_skip_human_won`` event, pulling
      ``human_class_validated``/``human_region_validated`` from the
      mget'd source for the audit trail; anything else -> ``errors``
      with ``phase: 'update'``.
    - A ``client.bulk()`` failure (transport/connection error) degrades
      to per-doc ``phase: 'update'`` error entries for that page rather
      than raising — callers already bare-except wrap this function
      (legacy_gemma.py, legacy_classes.py, legacy_auto_promote.py, bulk_writer.py),
      but a raise here would still lose partial progress on other pages.
    - ``refresh`` is forwarded to ``client.bulk(refresh=refresh)``
      unchanged (``True``/``False``/``'wait_for'``).

    Returns:
        ``{updated: int, skipped_due_to_conflict: int, errors: list[dict]}``
    """
    from src.clients.curation_opensearch import mget_crops

    updated = 0
    skipped = 0
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
                updated += 1
                continue
            error = action.get('error') or {}
            err_type = error.get('type', '')
            is_conflict = status == 409 or 'version_conflict' in err_type
            if is_conflict:
                logger.info(
                    'legacy_worker_skip_human_won',
                    doc_id=doc_id,
                    writer_id=writer_id,
                    human_class_validated=source.get('class_validated'),
                    human_region_validated=source.get(get_region_fields().validated),
                )
                skipped += 1
                continue
            errors.append({'doc_id': doc_id, 'phase': 'update', 'error': str(error or action)})

    return {'updated': updated, 'skipped_due_to_conflict': skipped, 'errors': errors}


def _is_human_marker(value: Any) -> bool:
    """A guard-field value indicates a human write iff it's a string
    containing the substring ``human``.

    Matches the in-codebase markers ``human``, ``human_move``, and
    ``vlm_human_confirmed`` (legacy_crops, legacy_plates, legacy_clustering).
    Non-human writers use ``ingest``, ``item_model``, ``coco_yolo11``,
    ``gemma``, ``cluster_majority_agreement``, etc.
    """
    return isinstance(value, str) and 'human' in value


def is_human_owned_class(source: dict[str, Any]) -> bool:
    """P0-2 reusable human-label guard predicate.

    True when a crop's current ``class_source`` indicates a human already
    set/confirmed the class label. Every automated CLASS writer must
    consult this before overwriting class fields — audit-remediation-plan
    Phase 1 (P0-2). Currently wired into:
      * the region-detection worker's classification gate (guards
        whether the combined VLM call is even asked to classify).
      * the region-detection worker's bulk writer (defense-in-depth on
        the write path — the classification gate already prevents class
        fields from reaching ``update_doc`` for a human crop in practice).
      * the VLM label-batch router endpoint (this endpoint takes
        caller-supplied ``crop_ids`` with no upstream query filter, so
        it's the site with no other protection).
      * the ingest pipeline router (defense-in-depth; the pipeline's own
        ``unvalidated_query`` already excludes ``class_validated=true``
        upstream).
    """
    return _is_human_marker(source.get('class_source'))


# Fields an automated class writer must never apply on top of a
# human-owned class row. Kept as a single source of truth so every
# consumer strips/skips the same surface — see ``is_human_owned_class``.
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

    Used by writers that may carry both class and plate fields in the
    same update (the SAM worker's combined path) so a human-label guard
    never leaks into suppressing an unrelated plate write — the
    LPR-scope requirement from Phase 1.
    """
    return {k: v for k, v in update.items() if k not in CLASS_WRITE_FIELDS}


# Companion fields preserved alongside a guard whenever the guard fires.
# When RegionFields.text_source says human, the matching
# RegionFields.text value was also human-set and must be preserved.
# Keep the map narrow — the guards themselves are the source of truth.
_HUMAN_GUARD_COMPANIONS: dict[str, tuple[str, ...]] = {
    get_region_fields().text_source: (get_region_fields().text,),
}

# Stricter than a companion: when the guard fires, the incoming value is
# never applied — the existing value is kept, or the field is left absent
# if the human-owned doc never had it. Class provenance describes who
# produced the *preserved* class_source, so an ingest detector's
# provenance must not land on (or be invented for) a human-owned row.
_HUMAN_GUARD_OWNED: dict[str, tuple[str, ...]] = {
    'class_source': (
        'class_detector',
        'class_detector_version',
        'class_labeler',
        'class_labeled_at',
    ),
}


async def occ_upsert_bulk(
    client: AsyncOpenSearch,
    docs: list[dict[str, Any]],
    *,
    index: str = ITEMS_INDEX,
    human_field_guards: list[str],
    writer_id: str = 'ingest',
    id_field: str = 'crop_id',
    refresh: bool | str = False,
    created_ids: list[str] | None = None,
    fill_if_absent: Collection[str] = (),
) -> dict[str, int]:
    """Upsert a batch of docs with OCC + human-label preservation.

    Replaces the blind ``opensearch.bulk(index=...)`` pattern used by
    ``ingest_one`` step 8. The blind pattern silently clobbered any
    human-applied label fields if a deterministic ``crop_id`` already
    existed (e.g. an operator deleted the images-index doc but the
    crop_id-stable items-index doc survived; or two parallel ingests
    slipped past the dedup).

    Decision tree per doc:

    1. ``mget`` all ids in one round-trip.
    2. Not present → ``op_type=create`` via bulk. On 409 (parallel writer
       beat us), fall back to the OCC update path with a re-fetch.
    3. Present, no human-write field set on the existing doc → OCC update
       with ``if_seq_no``/``if_primary_term``. On 409, retry once with a
       fresh fetch; a second 409 increments
       ``LEGACY_INGEST_OCC_FINAL_CONFLICT`` and the doc is skipped.
    4. Present, at least one human-guard field set → client-side merge:
       for each guarded field, if the existing doc already has it set
       (non-empty / non-None), keep the existing value; otherwise apply
       the new value. Increments ``LEGACY_INGEST_PRESERVED_HUMAN_LABEL``
       (labelled by which guard field fired) per preserved field — the
       Grafana proof-of-fix metric.
    5. Independently of 2-4, every ``fill_if_absent`` field is only a
       default for the *create*: on the update path it is applied only
       when the existing doc has no value for it, and otherwise dropped
       from the update so the existing value stands (e.g. a region
       status another writer already advanced).

    Args:
        client: AsyncOpenSearch instance.
        docs: New doc bodies. Each must contain ``id_field``.
        index: Target index.
        human_field_guards: Fields whose non-empty presence on the
            existing doc signals a human write to preserve.
        writer_id: Provenance tag for structured logs / metrics.
        id_field: Doc id field name (default ``crop_id``).
        refresh: OS refresh policy on the writes.
        created_ids: Optional out-list; the id of every doc this call
            newly created (bulk ``create`` acknowledged 200/201) is
            appended. Docs that already existed — including a create
            that lost a race and fell back to the update path — are not.
        fill_if_absent: Fields written on update only when the existing
            doc lacks them (see step 5).

    Returns:
        ``{'created': N, 'updated': M, 'preserved_human': P,
        'final_conflicts': C, 'filled_absent': F}`` — ``filled_absent``
        counts updated docs that received at least one
        ``fill_if_absent`` field.
    """
    # Local import: metrics module imports prometheus_client at top
    # level and we keep occ.py prometheus-free for unit-test ergonomics.
    from src.services.curation.metrics import (
        LEGACY_INGEST_OCC_FINAL_CONFLICT,
        LEGACY_INGEST_PRESERVED_HUMAN_LABEL,
    )

    result = {
        'created': 0,
        'updated': 0,
        'preserved_human': 0,
        'final_conflicts': 0,
        'filled_absent': 0,
    }
    if not docs:
        return result

    ids = [doc[id_field] for doc in docs]
    mget_resp = await client.mget(body={'ids': ids}, index=index)
    by_id: dict[str, dict[str, Any]] = {item['_id']: item for item in mget_resp.get('docs', [])}

    create_actions: list[dict[str, Any]] = []
    create_docs: list[dict[str, Any]] = []
    update_targets: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for doc in docs:
        doc_id = doc[id_field]
        existing = by_id.get(doc_id)
        if not existing or not existing.get('found'):
            create_actions.append({'create': {'_index': index, '_id': doc_id}})
            create_docs.append(doc)
        else:
            update_targets.append((doc, existing))

    # Phase 1: atomic creates for docs not present at mget time.
    create_conflict_ids: set[str] = set()
    if create_actions:
        body: list[dict[str, Any]] = []
        for action, doc in zip(create_actions, create_docs, strict=True):
            body.append(action)
            body.append(doc)
        bulk_resp = await client.bulk(body=body, refresh=refresh)
        for item in bulk_resp.get('items', []) or []:
            create_item = item.get('create') or {}
            status = create_item.get('status')
            doc_id = create_item.get('_id')
            if status in (200, 201):
                result['created'] += 1
                if created_ids is not None and doc_id:
                    created_ids.append(doc_id)
            elif status == 409 and doc_id:
                # A parallel writer created the doc between our mget and
                # our bulk create. Fall back to the OCC update path with
                # a fresh fetch.
                create_conflict_ids.add(doc_id)
            else:
                logger.warning(
                    'legacy_ingest_upsert_create_error',
                    doc_id=doc_id,
                    status=status,
                    error=create_item.get('error'),
                    writer_id=writer_id,
                )

    # Resolve create-conflicts by re-fetching and joining the update list.
    if create_conflict_ids:
        id_to_doc = {doc[id_field]: doc for doc in create_docs}
        refetch = await client.mget(body={'ids': sorted(create_conflict_ids)}, index=index)
        update_targets.extend(
            (id_to_doc[item['_id']], item)
            for item in refetch.get('docs', []) or []
            if item.get('found') and item['_id'] in id_to_doc
        )

    # Phase 2: per-doc OCC updates (single bulk would lose per-doc
    # if_seq_no/if_primary_term semantics — bulk update supports those
    # in newer OS, but per-doc keeps the conflict-retry logic clean).
    for new_doc, existing in update_targets:
        doc_id = new_doc[id_field]
        source = existing.get('_source') or {}
        seq_no = int(existing.get('_seq_no', 0))
        primary_term = int(existing.get('_primary_term', 1))

        merged, preserved_fields = _merge_preserving_human(
            new_doc=new_doc,
            existing=source,
            human_field_guards=human_field_guards,
        )
        filled = _apply_fill_if_absent(merged, source, fill_if_absent)

        attempt = 0
        max_retries = 1
        while True:
            try:
                await client.update(
                    index=index,
                    id=doc_id,
                    body={'doc': merged},
                    if_seq_no=seq_no,
                    if_primary_term=primary_term,
                    refresh=refresh,
                )
                result['updated'] += 1
                for field in preserved_fields:
                    # Per-field counter — Grafana panels can break down
                    # which guard fired (e.g. RegionFields.label_source vs
                    # class_source) so operators see the fix in action.
                    LEGACY_INGEST_PRESERVED_HUMAN_LABEL.labels(field=field).inc()
                result['preserved_human'] += len(preserved_fields)
                if filled:
                    result['filled_absent'] += 1
                break
            except Exception as exc:
                err_type = type(exc).__name__
                is_conflict = 'Conflict' in err_type or '409' in str(exc)
                if not is_conflict:
                    logger.warning(
                        'legacy_ingest_upsert_update_error',
                        doc_id=doc_id,
                        writer_id=writer_id,
                        error=str(exc),
                    )
                    break
                if attempt >= max_retries:
                    LEGACY_INGEST_OCC_FINAL_CONFLICT.inc()
                    result['final_conflicts'] += 1
                    logger.info(
                        'legacy_ingest_upsert_final_conflict',
                        doc_id=doc_id,
                        writer_id=writer_id,
                    )
                    break
                attempt += 1
                # Re-fetch to pick up the concurrent writer's changes,
                # then re-apply our merge so we never clobber the labeler.
                try:
                    fresh = await client.get(index=index, id=doc_id)
                except Exception as fetch_exc:
                    logger.warning(
                        'legacy_ingest_upsert_refetch_failed',
                        doc_id=doc_id,
                        error=str(fetch_exc),
                    )
                    break
                source = fresh.get('_source') or {}
                seq_no = int(fresh.get('_seq_no', 0))
                primary_term = int(fresh.get('_primary_term', 1))
                merged, preserved_fields = _merge_preserving_human(
                    new_doc=new_doc,
                    existing=source,
                    human_field_guards=human_field_guards,
                )
                filled = _apply_fill_if_absent(merged, source, fill_if_absent)

    return result


def _apply_fill_if_absent(
    merged: dict[str, Any],
    existing: dict[str, Any],
    fill_if_absent: Collection[str],
) -> bool:
    """Drop (in place) every ``fill_if_absent`` field the existing doc
    already has a value for. Returns whether any such field is still
    being written."""
    filled = False
    for field in fill_if_absent:
        if field not in merged:
            continue
        if existing.get(field) not in (None, ''):
            merged.pop(field)
        else:
            filled = True
    return filled


def _merge_preserving_human(
    *,
    new_doc: dict[str, Any],
    existing: dict[str, Any],
    human_field_guards: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Build the update body that preserves any human-set guard fields.

    A guard "fires" only when the existing doc's value matches the
    ``_is_human_marker`` predicate — i.e. a string containing ``human``.
    Non-human source values (``ingest``, ``item_model``, ``gemma``, etc.)
    do not trip preservation; ingest is free to overwrite them with its
    fresh-pass value.

    Returns ``(merged_doc, preserved_field_names)``. ``preserved_field_names``
    counts each guard that actually fired (used by the per-field Grafana
    counter to prove the fix exercises in practice).
    """
    merged = dict(new_doc)
    preserved: list[str] = []
    for field in human_field_guards:
        existing_val = existing.get(field)
        if _is_human_marker(existing_val):
            merged[field] = existing_val
            preserved.append(field)
            for companion in _HUMAN_GUARD_COMPANIONS.get(field, ()):
                companion_val = existing.get(companion)
                if companion_val not in (None, '', [], {}):
                    merged[companion] = companion_val
            for owned in _HUMAN_GUARD_OWNED.get(field, ()):
                if owned in existing:
                    merged[owned] = existing[owned]
                else:
                    merged.pop(owned, None)
    return merged, preserved


__all__ = [
    'ITEMS_INDEX',
    'Merger',
    'OCCFinalConflictError',
    'occ_skip_on_conflict_bulk',
    'occ_update_one',
    'occ_upsert_bulk',
]
