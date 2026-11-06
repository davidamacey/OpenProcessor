"""Write a bulk label or move as a recorded operation.

The record exists before the first item is written; each page's members are saved with the
state the write replaced as soon as the page is written. A crash leaves the record ``writing``
with its lease expired (served ``interrupted``), and the pages it wrote can still be undone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.occ import OCC_BULK_PAGE_SIZE
from src.core.logging import get_logger
from src.domain.job_lease import LeaseKind
from src.services.curation.crop_bulk import occ_bulk_human_relabel
from src.services.curation.label_ops.store import NewLabelOp, create_op, save_op, write_members
from src.services.curation.lease_heartbeat import leased
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

type Merger = Callable[[dict[str, Any]], dict[str, Any]]


async def recorded_relabel(
    opensearch: Any,
    ids: list[str],
    *,
    op: NewLabelOp,
    merger_for: Callable[[str], Merger],
) -> dict[str, Any]:
    """The batch relabel response plus ``op_id``. ``merger_for(op_id)`` builds the
    merge body that tags each item with the operation."""
    record = await create_op(op)
    priors: dict[str, Any] = {}
    merge_for_op = merger_for(record.op_id)

    def merge(source: dict[str, Any]) -> dict[str, Any]:
        """The operation's merge, keeping the state it replaced: the snapshot it appends."""
        merged = merge_for_op(source)
        history = merged.get('class_id_history') or []
        if history and history[-1].get('op_id') == record.op_id:
            priors[source['crop_id']] = history[-1]
        return merged

    updated_ids: list[str] = []
    conflicts: list[dict[str, Any]] = []
    async with leased(
        opensearch,
        LeaseKind.LABEL_OP,
        record.op_id,
        record.op_id,
        busy=lambda _holder: RuntimeError(f'label operation {record.op_id} is already writing'),
    ):
        # Page by page, recording each page's members with their prior state as soon as it is
        # written, so an operation that dies part-way can still be undone for the pages it wrote.
        for start in range(0, len(ids), OCC_BULK_PAGE_SIZE):
            result = await occ_bulk_human_relabel(
                opensearch, ids[start : start + OCC_BULK_PAGE_SIZE], merge, writer_id=op.writer
            )
            await write_members(
                record, [(i, priors[i]) for i in result['updated_ids'] if i in priors]
            )
            priors.clear()
            updated_ids += result['updated_ids']
            conflicts += result['conflicts']
        result = {'updated': len(updated_ids), 'updated_ids': updated_ids, 'conflicts': conflicts}
        record.status = 'written'
        record.count = result['updated']
        record.conflicts = len(result['conflicts'])
        record.finished_at = now_iso()
        await save_op(record)
    logger.info(
        'label_op_written', op_id=record.op_id, count=record.count, conflicts=record.conflicts
    )
    return {**result, 'op_id': record.op_id}
