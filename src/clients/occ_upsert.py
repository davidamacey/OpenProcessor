"""Ingest upsert with OCC and human-label preservation.

Split out of :mod:`src.clients.occ` to keep that module under the 700-LOC ratchet, a
real seam: ``occ.py`` holds single-document and worker-semantics OCC, ``occ_bulk.py`` the
human-write batch update, and this module the create-or-merge upsert that ingest, dataset
import and re-detection write item documents with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.clients.occ import OCC_BULK_MGET_SOURCE_EXCLUDES
from src.clients.occ_locks import _is_locked_marker, is_locked_class
from src.config.curation import items_index
from src.core.logging import get_logger
from src.services.curation.embedding_state import keep_stored_vector_state
from src.services.curation.history import CLASS_STATE_FIELDS


if TYPE_CHECKING:
    from collections.abc import Collection

    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)


@dataclass
class DocOutcome:
    """What one upsert did to one document, for callers that report per image.

    ``status`` is ``created``, ``updated``, ``conflict`` (OCC retries exhausted) or ``error``
    (the store refused the document or the refetch failed); ``error`` carries the reason.
    """

    status: str
    preserved: int = 0
    filled: bool = False
    error: str | None = None


# When a guard fires, the incoming value is
# never applied — the existing value is kept, or the field is left absent
# if the human-owned doc never had it. Class provenance describes who
# produced the *preserved* class_source, so an ingest detector's
# provenance must not land on (or be invented for) a human-owned row.
_HUMAN_GUARD_OWNED: dict[str, tuple[str, ...]] = {
    'class_source': (*CLASS_STATE_FIELDS, 'test_holdout'),
}


async def occ_upsert_bulk(
    client: AsyncOpenSearch,
    docs: list[dict[str, Any]],
    *,
    index: str | None = None,
    human_field_guards: list[str],
    writer_id: str = 'ingest',
    id_field: str = 'crop_id',
    refresh: bool | str = False,
    created_ids: list[str] | None = None,
    fill_if_absent: Collection[str] = (),
    outcomes: dict[str, DocOutcome] | None = None,
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
       ``OP_INGEST_OCC_FINAL_CONFLICT`` and the doc is skipped.
    4. Present, at least one human-guard field set → client-side merge:
       for each guarded field, if the existing doc already has it set
       (non-empty / non-None), keep the existing value; otherwise apply
       the new value. Increments ``OP_INGEST_PRESERVED_HUMAN_LABEL``
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
        outcomes: Optional out-dict; receives one :class:`DocOutcome` per
            doc id, so a caller that batched several images' documents into
            this call can report each image on its own.

    Returns:
        ``{'created': N, 'updated': M, 'preserved_human': P,
        'final_conflicts': C, 'filled_absent': F}`` — ``filled_absent``
        counts updated docs that received at least one
        ``fill_if_absent`` field.
    """
    if index is None:
        index = items_index()
    # Local import: metrics module imports prometheus_client at top
    # level and we keep occ.py prometheus-free for unit-test ergonomics.
    from src.services.curation.metrics import (
        OP_INGEST_OCC_FINAL_CONFLICT,
        OP_INGEST_PRESERVED_HUMAN_LABEL,
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
    mget_resp = await client.mget(
        body={'ids': ids}, index=index, _source_excludes=OCC_BULK_MGET_SOURCE_EXCLUDES
    )
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
                if outcomes is not None and doc_id:
                    outcomes[doc_id] = DocOutcome('created')
                if created_ids is not None and doc_id:
                    created_ids.append(doc_id)
            elif status == 409 and doc_id:
                # A parallel writer created the doc between our mget and
                # our bulk create. Fall back to the OCC update path with
                # a fresh fetch.
                create_conflict_ids.add(doc_id)
            else:
                logger.warning(
                    'curation_ingest_upsert_create_error',
                    doc_id=doc_id,
                    status=status,
                    error=create_item.get('error'),
                    writer_id=writer_id,
                )
                if outcomes is not None and doc_id:
                    outcomes[doc_id] = DocOutcome('error', error=_item_error(create_item))

    # Resolve create-conflicts by re-fetching and joining the update list.
    if create_conflict_ids:
        id_to_doc = {doc[id_field]: doc for doc in create_docs}
        refetch = await client.mget(
            body={'ids': sorted(create_conflict_ids)},
            index=index,
            _source_excludes=OCC_BULK_MGET_SOURCE_EXCLUDES,
        )
        update_targets.extend(
            (id_to_doc[item['_id']], item)
            for item in refetch.get('docs', []) or []
            if item.get('found') and item['_id'] in id_to_doc
        )

    # Phase 2: batched OCC updates — one occ_update_bulk call
    # (one mget page + one bulk, retrying 409s once) instead of one
    # client.update per doc. Local import: occ_bulk.py imports FROM this
    # module (OCC_BULK_*), so importing it back at module
    # level here would cycle.
    if update_targets:
        from src.clients.occ_bulk import occ_update_bulk

        new_doc_by_id = {doc[id_field]: doc for doc, _existing in update_targets}
        # merge_fn is re-invoked per retry round with the freshest source,
        # so this always reflects the merge that actually got written (or
        # was last attempted) for each id.
        merge_effects: dict[str, tuple[list[str], bool]] = {}

        def _merge(doc_id: str, source: dict[str, Any]) -> dict[str, Any]:
            merged, preserved_fields = _merge_preserving_human(
                new_doc=new_doc_by_id[doc_id],
                existing=source,
                human_field_guards=human_field_guards,
            )
            filled = _apply_fill_if_absent(merged, source, fill_if_absent)
            keep_stored_vector_state(merged, source)
            merge_effects[doc_id] = (preserved_fields, filled)
            return merged

        try:
            status_map = await occ_update_bulk(
                client,
                index=index,
                ids=list(new_doc_by_id),
                merge_fn=_merge,
                max_retries=1,
                refresh=refresh,
            )
        except Exception as exc:
            logger.warning(
                'curation_ingest_upsert_bulk_update_failed',
                writer_id=writer_id,
                n=len(new_doc_by_id),
                error=str(exc),
            )
            status_map = {}

        for doc_id in new_doc_by_id:
            status = status_map.get(doc_id)
            if status == 'updated':
                preserved_fields, filled = merge_effects.get(doc_id, ([], False))
                result['updated'] += 1
                if outcomes is not None:
                    outcomes[doc_id] = DocOutcome('updated', len(preserved_fields), filled)
                for field in preserved_fields:
                    # Per-field counter — Grafana panels can break down which guard fired
                    # (e.g. RegionFields.label_source vs class_source) so operators see the fix.
                    OP_INGEST_PRESERVED_HUMAN_LABEL.labels(field=field).inc()
                result['preserved_human'] += len(preserved_fields)
                if filled:
                    result['filled_absent'] += 1
            elif status == 'conflict-exhausted':
                # occ_update_bulk folds genuine (non-conflict) item errors into this
                # same status — logged distinctly there via curation_occ_update_bulk_item_error.
                OP_INGEST_OCC_FINAL_CONFLICT.inc()
                result['final_conflicts'] += 1
                if outcomes is not None:
                    outcomes[doc_id] = DocOutcome('conflict')
                logger.info(
                    'curation_ingest_upsert_final_conflict', doc_id=doc_id, writer_id=writer_id
                )
            else:
                logger.warning('curation_ingest_upsert_refetch_failed', doc_id=doc_id, error=status)
                if outcomes is not None:
                    outcomes[doc_id] = DocOutcome('error', error=str(status or 'update failed'))

    return result


def _item_error(item: dict[str, Any]) -> str:
    """A bulk item's refusal as one line: ``status: type: reason``."""
    error = item.get('error')
    if isinstance(error, dict):
        return f'{item.get("status")}: {error.get("type")}: {error.get("reason")}'
    return f'{item.get("status")}: {error}'


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


# Guards that describe the item's CLASS state (as opposed to a region-
# level field like region_text_source) must fire on the same contract
# is_locked_class() defines, not a narrower per-value string check --
# otherwise re-ingest preserves an unvalidated ("suggestion") import's
# class_source/label_source the same as a human's or a VALIDATED
# import's, which is_locked_class explicitly does NOT lock. Region-only
# guard fields (region_label_source, region_text_source, ...) keep the
# narrower per-value check -- is_locked_class doesn't speak to them.
_CLASS_LOCK_GUARD_FIELDS = frozenset({'class_source', 'label_source'})


def _merge_preserving_human(
    *,
    new_doc: dict[str, Any],
    existing: dict[str, Any],
    human_field_guards: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Build the update body that preserves any human- or import-set
    guard fields.

    For ``class_source``/``label_source``, a guard "fires" exactly when
    :func:`is_locked_class` says the item's class is locked (human-set,
    a VALIDATED import, or ``test_holdout``) -- matching that contract
    means an unvalidated ("suggestion") import's provenance is free to
    be overwritten by a fresh ingest pass, same as ``ingest``/``vlm``/etc.
    Every other guard field keeps the narrower :func:`_is_locked_marker`
    per-value check (a string containing ``human``, or an import
    provenance marker) -- ``is_locked_class`` doesn't speak to
    region-level fields.

    Returns ``(merged_doc, preserved_field_names)``. ``preserved_field_names``
    counts each guard that actually fired (used by the per-field Grafana
    counter to prove the fix exercises in practice).
    """
    merged = dict(new_doc)
    preserved: list[str] = []
    for field in human_field_guards:
        existing_val = existing.get(field)
        fires = (
            is_locked_class(existing)
            if field in _CLASS_LOCK_GUARD_FIELDS
            else _is_locked_marker(existing_val)
        )
        if fires:
            merged[field] = existing_val
            preserved.append(field)
            for owned in _HUMAN_GUARD_OWNED.get(field, ()):
                if owned in existing:
                    merged[owned] = existing[owned]
                else:
                    merged.pop(owned, None)
    return merged, preserved


__all__ = ['DocOutcome', 'occ_upsert_bulk']
