"""Write a bulk label or move as a recorded operation.

The record exists before the first item is written; the ids written are saved
after the bulk returns. A crash in between leaves the record ``writing`` with
its lock free (served ``interrupted``), and the items it did write are still
found through their ``label_op_id`` tag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.curation.crop_bulk import occ_bulk_human_relabel
from src.services.curation.label_ops.store import (
    NewLabelOp,
    create_op,
    save_op,
    write_lock,
    write_members,
)
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
    record = create_op(op)
    with write_lock(record.op_id) as got:
        if not got:  # a fresh dir: someone else cannot hold its lock
            raise RuntimeError(f'label operation {record.op_id} is already writing')
        result = await occ_bulk_human_relabel(
            opensearch, ids, merger_for(record.op_id), writer_id=op.writer
        )
        try:
            write_members(record.op_id, result['updated_ids'])
        except OSError as exc:
            logger.error('label_op_members_failed', op_id=record.op_id, error=str(exc))
        record.status = 'written'
        record.count = result['updated']
        record.conflicts = len(result['conflicts'])
        record.finished_at = now_iso()
        save_op(record)
    logger.info(
        'label_op_written', op_id=record.op_id, count=record.count, conflicts=record.conflicts
    )
    return {**result, 'op_id': record.op_id}
