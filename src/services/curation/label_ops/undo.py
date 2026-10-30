"""Undo a label operation: restore, per item, what the operation wrote.

Each item's outcome is decided from the item itself (:func:`~src.domain.label_ops.undo_outcome`),
inside the OCC merge, so a person's write that races the undo wins and re-running after
any failure restores exactly the remainder. One paged read-merge-conditional-bulk is in
flight at a time; the pages are those of :func:`~src.clients.occ_bulk.occ_bulk_rounds`.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.crops import mget_crops
from src.clients.occ import OCC_BULK_PAGE_SIZE
from src.clients.occ_bulk import occ_bulk_rounds
from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.domain.label_ops import UndoOutcome, undo_outcome
from src.services.curation.class_undo import UNDO_READ_FIELDS, class_undo_update
from src.services.curation.item_selection import MAX_BULK_SELECTION
from src.services.curation.label_ops.store import (
    UNDO_PASSES_KEPT,
    LabelOpRecord,
    ServedStatus,
    read_members,
    read_op,
    save_op,
    served_status,
    undo_lock,
    write_lock,
)
from src.services.curation.label_ops.views import current_counts
from src.services.curation.reprocess_targets import scan_items
from src.services.rejections import rejection
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

_PASS_KEYS = (
    'restored',
    'already_undone',
    'kept_changed',
    'kept_holdout',
    'kept_stale_class',
    'conflicts',
)
CONFLICT = 'conflict'
_PENDING = 'pending'
_CONFLICT_IDS_SHOWN = 100


async def _members(opensearch: Any, record: LabelOpRecord) -> tuple[list[str], bool]:
    """The ids the operation wrote; from the tag when ``members.json`` is missing."""
    known = read_members(record.op_id)
    if known is not None:
        return known, True
    hits = await scan_items(
        opensearch,
        {'term': {'label_op_id': record.op_id}},
        index=items_index(),
        includes=['crop_id'],
        max_docs=MAX_BULK_SELECTION + 1,
    )
    if len(hits) > MAX_BULK_SELECTION:
        logger.error('label_op_tag_overflow', op_id=record.op_id, limit=MAX_BULK_SELECTION)
        msg = f'label operation {record.op_id} tags more than {MAX_BULK_SELECTION} items'
        raise RuntimeError(msg)
    return [doc_id for doc_id, _ in hits], False


async def _classify(
    opensearch: Any,
    record: LabelOpRecord,
    ids: list[str],
    registry_name: Callable[[int], str | None],
) -> dict[str, str]:
    """Dry run: the outcome per id, read in pages, nothing written."""
    outcomes: dict[str, str] = {}
    for start in range(0, len(ids), OCC_BULK_PAGE_SIZE):
        page = ids[start : start + OCC_BULK_PAGE_SIZE]
        docs = await mget_crops(
            opensearch, page, index=items_index(), source_includes=UNDO_READ_FIELDS
        )
        for doc_id in page:
            source = (docs.get(doc_id) or {}).get('_source')
            outcomes[doc_id] = undo_outcome(
                source, record.fingerprint, registry_name=registry_name
            ).value
    return outcomes


async def _restore(
    opensearch: Any,
    record: LabelOpRecord,
    ids: list[str],
    registry_name: Callable[[int], str | None],
) -> dict[str, str]:
    outcomes: dict[str, str] = {}

    def prepare(doc_id: str, doc: dict[str, Any] | None) -> dict[str, Any] | None:
        source = None if doc is None else doc.get('_source') or {}
        outcome = undo_outcome(source, record.fingerprint, registry_name=registry_name)
        if outcome is not UndoOutcome.RESTORE:
            outcomes[doc_id] = outcome.value
            return None
        assert source is not None
        outcomes[doc_id] = _PENDING  # overwritten, never incremented, on a retry round
        return class_undo_update(source)

    def on_written(doc_id: str) -> None:
        outcomes[doc_id] = UndoOutcome.RESTORE.value

    def on_failed(doc_id: str, _doc: dict[str, Any] | None, _why: str) -> None:
        outcomes[doc_id] = CONFLICT

    await occ_bulk_rounds(
        opensearch,
        index=items_index(),
        ids=ids,
        prepare=prepare,
        on_written=on_written,
        on_failed=on_failed,
        max_retries=5,
        refresh='wait_for',
        source_includes=UNDO_READ_FIELDS,
    )
    return outcomes


def _response(
    record: LabelOpRecord,
    outcomes: dict[str, str],
    *,
    dry_run: bool,
    members_known: bool,
    status: ServedStatus,
) -> dict[str, Any]:
    tally = Counter(outcomes.values())
    return {
        'op_id': record.op_id,
        'dry_run': dry_run,
        'status': status,
        'members_known': members_known,
        'matched': len(outcomes),
        'restored': tally[UndoOutcome.RESTORE.value],
        'already_undone': tally[UndoOutcome.ALREADY_UNDONE.value],
        'kept_changed': tally[UndoOutcome.KEPT_CHANGED.value],
        'kept_holdout': tally[UndoOutcome.KEPT_HOLDOUT.value],
        'kept_stale_class': tally[UndoOutcome.KEPT_STALE_CLASS.value],
        'conflicts': tally[CONFLICT],
        'conflict_ids': sorted(i for i, o in outcomes.items() if o == CONFLICT)[
            :_CONFLICT_IDS_SHOWN
        ],
    }


async def _refuse_if_nothing_left(opensearch: Any, record: LabelOpRecord) -> None:
    """``409 already_undone``: an undo pass finished and no item still carries the label."""
    if record.status != 'undone':
        return
    if (await current_counts(opensearch, [record]))[record.op_id] == 0:
        raise rejection(409, 'already_undone', f'{record.op_id} was already undone')


async def undo_label_op(
    opensearch: Any,
    op_id: str,
    *,
    dry_run: bool,
    registry_name: Callable[[int], str | None],
) -> dict[str, Any]:
    """The :class:`LabelOpUndoResponse` of an undo pass over the operation."""
    if dry_run:
        record = read_op(op_id)
        await _refuse_if_nothing_left(opensearch, record)
        ids, known = await _members(opensearch, record)
        outcomes = await _classify(opensearch, record, ids, registry_name)
        return _response(
            record, outcomes, dry_run=True, members_known=known, status=served_status(record)
        )
    read_op(op_id)  # 404 before any lock file is created
    with undo_lock() as got:
        if not got:
            raise rejection(409, 'label_op_busy', 'another undo is running in this project')
        record = read_op(op_id)  # fresh: a pass that just finished changed it
        with write_lock(op_id) as writing:
            if not writing:
                raise rejection(409, 'label_op_in_progress', f'{op_id} is still being written')
            await _refuse_if_nothing_left(opensearch, record)
            ids, known = await _members(opensearch, record)
            record.status = 'undoing'
            save_op(record)
            outcomes = await _restore(opensearch, record, ids, registry_name)
            response = _response(
                record, outcomes, dry_run=False, members_known=known, status='undone'
            )
            record.status = 'undone'
            record.undo_passes = [
                *record.undo_passes,
                {
                    'at': now_iso(),
                    **{k: response[k] for k in _PASS_KEYS},
                },
            ][-UNDO_PASSES_KEPT:]
            save_op(record)
    logger.info('label_op_undone', op_id=op_id, **{k: response[k] for k in _PASS_KEYS})
    return response
