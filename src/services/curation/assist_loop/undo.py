"""Undo what one auto-label loop propagated: the label-operation undo over its operations.

Every propagated node is one recorded label operation (kind ``assist_loop_propagate``, writer
``assist_loop:<loop_id>``), so rolling a loop back is the undo every label operation has: an
item is restored only while it is exactly as the operation left it, and anything changed since,
locked, validated, excluded or in the holdout is kept and counted. Restoring is idempotent.

The operations are the loop's by writer, not only the ones its node rows link: a slice that
died inside a propagation wrote an operation its node row never learned of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.exceptions import RequestRejectedError
from src.services.curation.label_ops.store import list_ops
from src.services.curation.label_ops.undo import undo_label_op
from src.services.curation.registry_access import current_registry


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_OPS_SCANNED = 50
_COUNTS = ('matched', 'restored', 'kept_changed', 'conflicts')


@dataclass
class UndoResult:
    dry_run: bool
    matched: int = 0
    """Items the loop wrote that no rollback restored yet."""
    restored: int = 0
    """Restored, or that a dry run would restore."""
    kept_changed: int = 0
    conflicts: int = 0
    ops: list[str] = field(default_factory=list)


async def loop_op_ids(
    loop_id: str, links: Mapping[str, str], node_paths: Sequence[str] | None
) -> list[str]:
    """The label operations of the loop (of ``node_paths`` when given), oldest first.
    ``links`` maps a node path to the operation its propagation was recorded as."""
    if node_paths is not None:
        return sorted({links[p] for p in node_paths if p in links})
    writer = f'assist_loop:{loop_id}'
    by_writer = {r.op_id for r in await list_ops(limit=_OPS_SCANNED) if r.writer == writer}
    return sorted({*links.values(), *by_writer})


async def undo_loop(
    opensearch: Any,
    *,
    loop_id: str,
    links: Mapping[str, str],
    node_paths: Sequence[str] | None = None,
    dry_run: bool = True,
) -> UndoResult:
    """Undo the loop's operations and add their counts."""
    names = {c.class_id: c.class_name for c in (await current_registry()).classes}
    result = UndoResult(dry_run=dry_run)
    for op_id in await loop_op_ids(loop_id, links, node_paths):
        try:
            one = await undo_label_op(opensearch, op_id, dry_run=dry_run, registry_name=names.get)
        except RequestRejectedError as refusal:
            if getattr(refusal, 'detail', {}).get('error') == 'already_undone':
                continue
            raise
        result.ops.append(op_id)
        for key in _COUNTS:
            setattr(result, key, getattr(result, key) + int(one[key]))
    return result


__all__ = ['UndoResult', 'loop_op_ids', 'undo_loop']
