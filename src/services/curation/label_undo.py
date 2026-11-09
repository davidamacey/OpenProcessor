"""Undo of human class writes, the other recorded per-item decisions (discard,
VLM-suggestion dismissal, review un-dismissal) and the item's class history.

Every write is an OCC read-merge-write (``src.clients.occ``) whose merger never touches a
locked field the lock rule protects: the mergers here only restore a recorded class state
or stamp the decision's own fields. Failures are typed: :class:`CropNotFoundError` only for
a missing document; an OpenSearch outage propagates as the client's own exception so the
router can map it to its outage response instead of reporting "not found".
"""

from __future__ import annotations

from typing import Any

from opensearchpy.exceptions import NotFoundError

from src.clients.occ import Merger, occ_update_one
from src.clients.occ_bulk import occ_update_bulk
from src.config.curation_accessors import items_index
from src.domain.history import (
    CLASS_STATE_FIELDS,
    HUMAN_DISCARD_WRITER,
    HUMAN_UNLABEL_WRITER,
    REVIEW_DISMISS_FIELDS,
    find_undo_snapshot,
    record_class_snapshot,
)
from src.services.curation.class_sources import vlm_suggestion
from src.services.curation.class_undo import class_undo_update
from src.services.curation.edit_history import EDIT_HISTORY_FIELD, EditKind, record_edit
from src.services.curation.wire import item_source_excludes, serialize_item
from src.services.embedding_spaces.readers import item_state_fields, vector_source_excludes
from src.utils.timeutil import now_iso


VLM_DISMISS_WRITER = 'human:vlm_dismiss'
REVIEW_UNDISMISS_WRITER = 'human:review_undismiss'

HISTORY_KEYS: tuple[str, ...] = (
    *CLASS_STATE_FIELDS,
    *REVIEW_DISMISS_FIELDS,
    'op_id',
    'writer',
    'at',
)


class CropNotFoundError(LookupError):
    """The item document does not exist."""

    def __init__(self, crop_id: str) -> None:
        super().__init__(crop_id)
        self.crop_id = crop_id


class NothingToUndoError(Exception):
    """The item has no un-undone human class write on record."""


class NoVlmSuggestionError(Exception):
    """The item has no VLM class suggestion to dismiss."""


async def _occ_write(opensearch: Any, crop_id: str, merger: Merger, *, writer_id: str) -> None:
    """One OCC write of ``merger`` to the item. Raises :class:`CropNotFoundError` for a missing
    document, ``OCCFinalConflictError`` when the retries run out; the merger's own typed
    errors and any OpenSearch failure propagate unchanged."""
    try:
        await occ_update_one(
            opensearch,
            vector_excludes=await vector_source_excludes(),
            doc_id=crop_id,
            merger=merger,
            refresh='wait_for',
            writer_id=writer_id,
        )
    except NotFoundError as exc:
        raise CropNotFoundError(crop_id) from exc


async def items_by_ids(opensearch: Any, ids: list[str]) -> list[dict[str, Any]]:
    """The wire items for the ids that exist, in the order OpenSearch answers."""
    if not ids:
        return []
    resp = await opensearch.mget(
        index=items_index(),
        body={'ids': ids},
        _source_excludes=item_source_excludes(await vector_source_excludes()),
    )
    state_fields = await item_state_fields()
    return [
        serialize_item(d.get('_source') or {}, d.get('_id', ''), state_fields=state_fields)
        for d in (resp.get('docs') or [])
        if d.get('found')
    ]


async def item_by_id(opensearch: Any, crop_id: str) -> dict[str, Any]:
    """The wire item, or :class:`CropNotFoundError`."""
    items = await items_by_ids(opensearch, [crop_id])
    if not items:
        raise CropNotFoundError(crop_id)
    return items[0]


def undo_merger(*, require_history: bool) -> Merger:
    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        if require_history and find_undo_snapshot(current.get('class_id_history')) is None:
            raise NothingToUndoError
        return class_undo_update(current)

    return _merge


async def undo_one(opensearch: Any, crop_id: str, *, require_history: bool) -> None:
    """Restore one item to its state before its newest human class write.

    Raises :class:`NothingToUndoError` (only with ``require_history``),
    :class:`CropNotFoundError` or ``OCCFinalConflictError``."""
    await _occ_write(
        opensearch,
        crop_id,
        undo_merger(require_history=require_history),
        writer_id=HUMAN_UNLABEL_WRITER,
    )


def discard_merger(*, clear_class: bool, dismiss_from_review: bool) -> Merger:
    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        now = now_iso()
        update: dict[str, Any] = {
            'class_id_history': record_class_snapshot(
                current, writer=HUMAN_DISCARD_WRITER, restorable=True
            ),
            'updated_at': now,
        }
        if clear_class:
            update.update(dict.fromkeys(CLASS_STATE_FIELDS))
            update['class_validated'] = False
        if dismiss_from_review:
            update['review_dismissed_at'] = now
            update['review_dismissed_by'] = 'human'
        return update

    return _merge


async def discard_one(
    opensearch: Any, crop_id: str, *, clear_class: bool, dismiss_from_review: bool
) -> None:
    await _occ_write(
        opensearch,
        crop_id,
        discard_merger(clear_class=clear_class, dismiss_from_review=dismiss_from_review),
        writer_id=HUMAN_DISCARD_WRITER,
    )


async def discard_many(
    opensearch: Any, crop_ids: list[str], *, clear_class: bool, dismiss_from_review: bool
) -> dict[str, list[str]]:
    """Discard each item in one conditional bulk; the ids per outcome: ``discarded``,
    ``conflicts`` (a concurrent write outlasted the retries) and ``not_found``."""
    merge = discard_merger(clear_class=clear_class, dismiss_from_review=dismiss_from_review)
    status = await occ_update_bulk(
        opensearch,
        vector_excludes=await vector_source_excludes(),
        index=items_index(),
        ids=crop_ids,
        merge_fn=lambda _crop_id, source: merge(source),
        max_retries=5,
        refresh='wait_for',
    )
    by: dict[str, list[str]] = {'discarded': [], 'conflicts': [], 'not_found': []}
    outcome_of = {'updated': 'discarded', 'not-found': 'not_found'}
    for crop_id in crop_ids:
        by[outcome_of.get(status.get(crop_id, ''), 'conflicts')].append(crop_id)
    return by


def vlm_dismiss_merger(current: dict[str, Any]) -> dict[str, Any]:
    class_id, class_name = vlm_suggestion(current)
    if class_name is None:
        raise NoVlmSuggestionError
    return {
        'vlm_dismissed_class_id': class_id,
        'vlm_dismissed_class_name': class_name,
        'vlm_dismissed_at': now_iso(),
        EDIT_HISTORY_FIELD: record_edit(
            current, kind=EditKind.VLM_DISMISS, writer=VLM_DISMISS_WRITER
        ),
        'updated_at': now_iso(),
    }


async def dismiss_vlm_suggestion(opensearch: Any, crop_id: str) -> None:
    """Record the rejection of the item's VLM class suggestion. Raises
    :class:`NoVlmSuggestionError` when there is none."""
    await _occ_write(opensearch, crop_id, vlm_dismiss_merger, writer_id=VLM_DISMISS_WRITER)


def _review_undismiss_merger(_current: dict[str, Any]) -> dict[str, Any]:
    return {'review_dismissed_at': None, 'review_dismissed_by': None, 'updated_at': now_iso()}


async def review_undismiss(opensearch: Any, crop_id: str) -> None:
    """Clear the item's review dismissal (the class is untouched)."""
    await _occ_write(
        opensearch, crop_id, _review_undismiss_merger, writer_id=REVIEW_UNDISMISS_WRITER
    )


async def crop_history(opensearch: Any, crop_id: str) -> list[dict[str, Any]]:
    """The item's class history entries, oldest first, with only the documented keys."""
    try:
        resp = await opensearch.get(
            index=items_index(), id=crop_id, _source_includes=['class_id_history']
        )
    except NotFoundError as exc:
        raise CropNotFoundError(crop_id) from exc
    history = (resp.get('_source') or {}).get('class_id_history') or []
    return [{k: entry.get(k) for k in HISTORY_KEYS} for entry in history if isinstance(entry, dict)]
