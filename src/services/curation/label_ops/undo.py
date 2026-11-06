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
from src.domain.job_lease import LeaseKind
from src.domain.label_ops import UndoOutcome, undo_outcome
from src.services.curation.class_undo import UNDO_READ_FIELDS, restore_update
from src.services.curation.label_ops.store import (
    UNDO_PASSES_KEPT,
    LabelOpRecord,
    ServedStatus,
    member_pages,
    read_op,
    save_op,
    served_status,
)
from src.services.curation.label_ops.views import current_counts
from src.services.curation.lease_heartbeat import leased
from src.services.rejections import rejection
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.db.repos.records import LabelOperationMemberLoad

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


async def _classify(
    opensearch: Any,
    record: LabelOpRecord,
    members: list[LabelOperationMemberLoad],
    registry_name: Callable[[int], str | None],
) -> dict[str, str]:
    """Dry run: the outcome per id, read in pages, nothing written."""
    priors = {m.item_id: m.prior for m in members}
    ids = list(priors)
    outcomes: dict[str, str] = {}
    for start in range(0, len(ids), OCC_BULK_PAGE_SIZE):
        page = ids[start : start + OCC_BULK_PAGE_SIZE]
        docs = await mget_crops(
            opensearch, page, index=items_index(), source_includes=UNDO_READ_FIELDS
        )
        for doc_id in page:
            source = (docs.get(doc_id) or {}).get('_source')
            outcomes[doc_id] = undo_outcome(
                source, record.fingerprint, prior=priors[doc_id], registry_name=registry_name
            ).value
    return outcomes


async def _restore(
    opensearch: Any,
    record: LabelOpRecord,
    members: list[LabelOperationMemberLoad],
    registry_name: Callable[[int], str | None],
) -> dict[str, str]:
    priors = {m.item_id: m.prior for m in members}
    outcomes: dict[str, str] = {}

    def prepare(doc_id: str, doc: dict[str, Any] | None) -> dict[str, Any] | None:
        source = None if doc is None else doc.get('_source') or {}
        outcome = undo_outcome(
            source, record.fingerprint, prior=priors[doc_id], registry_name=registry_name
        )
        if outcome is not UndoOutcome.RESTORE:
            outcomes[doc_id] = outcome.value
            return None
        assert source is not None
        outcomes[doc_id] = _PENDING  # overwritten, never incremented, on a retry round
        return restore_update(source, priors[doc_id])

    def on_written(doc_id: str) -> None:
        outcomes[doc_id] = UndoOutcome.RESTORE.value

    def on_failed(doc_id: str, _doc: dict[str, Any] | None, _why: str) -> None:
        outcomes[doc_id] = CONFLICT

    await occ_bulk_rounds(
        opensearch,
        index=items_index(),
        ids=list(priors),
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
    status: ServedStatus,
) -> dict[str, Any]:
    tally = Counter(outcomes.values())
    return {
        'op_id': record.op_id,
        'dry_run': dry_run,
        'status': status,
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
        record = await read_op(op_id)
        await _refuse_if_nothing_left(opensearch, record)
        outcomes: dict[str, str] = {}
        async for members in member_pages(record):
            outcomes.update(await _classify(opensearch, record, members, registry_name))
        return _response(
            record, outcomes, dry_run=True, status=await served_status(opensearch, record)
        )
    await read_op(op_id)  # 404 before any lease is taken
    async with (
        leased(
            opensearch,
            LeaseKind.LABEL_UNDO,
            None,
            op_id,
            busy=lambda _h: rejection(
                409, 'label_op_busy', 'another undo is running in this project'
            ),
        ),
        leased(
            opensearch,
            LeaseKind.LABEL_OP,
            op_id,
            op_id,
            busy=lambda _h: rejection(
                409, 'label_op_in_progress', f'{op_id} is still being written'
            ),
        ),
    ):
        record = await read_op(op_id)  # fresh: a pass that just finished changed it
        await _refuse_if_nothing_left(opensearch, record)
        record.status = 'undoing'
        await save_op(record)
        outcomes = {}
        async for members in member_pages(record):
            outcomes.update(await _restore(opensearch, record, members, registry_name))
        response = _response(record, outcomes, dry_run=False, status='undone')
        record.status = 'undone'
        record.undo_passes = [
            *record.undo_passes,
            {
                'at': now_iso(),
                **{k: response[k] for k in _PASS_KEYS},
            },
        ][-UNDO_PASSES_KEPT:]
        await save_op(record)
    logger.info('label_op_undone', op_id=op_id, **{k: response[k] for k in _PASS_KEYS})
    return response
