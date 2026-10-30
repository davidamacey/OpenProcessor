"""Curation router sub-module — recorded bulk label and move operations."""

from __future__ import annotations

from typing import Any

from fastapi import Query

from src.routers.curation._common import OpenSearchDep, RegistryDep, router
from src.schemas.curation.label_ops import LabelOpList, LabelOpUndoRequest, LabelOpUndoResponse
from src.services.curation.label_ops.undo import undo_label_op
from src.services.curation.label_ops.views import list_label_ops


@router.get('/label_ops', response_model=LabelOpList)
async def get_label_ops(
    opensearch: OpenSearchDep,
    limit: int = Query(default=20, ge=1, le=50),
    undoable: bool | None = Query(default=None, description='true: only undoable operations.'),
) -> dict[str, Any]:
    """The project's recorded bulk label and move operations, newest first.

    ``current`` is how many items still carry the operation's label; an
    operation is ``undoable`` while that is above zero and it is not being
    written or undone. Class ids are never served: identity is the name.
    """
    return await list_label_ops(opensearch, limit=limit, undoable_only=bool(undoable))


@router.post(
    '/label_ops/{op_id}/undo',
    response_model=LabelOpUndoResponse,
    responses={
        404: {'description': 'label_op_not_found: unknown, malformed or pruned operation.'},
        409: {
            'description': 'already_undone, label_op_busy (another undo is running) or '
            'label_op_in_progress (the operation is still being written).'
        },
    },
)
async def undo_op(
    op_id: str,
    payload: LabelOpUndoRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
) -> dict[str, Any]:
    """Undo a bulk label or move, item by item, from the stored operation.

    An item is restored only while it still carries the operation's label and
    nothing changed it since: a later edit, an exclusion, a class merge or the
    test holdout keeps the item as it is, and a restore target whose class was
    renamed is refused (``kept_stale_class``). ``dry_run`` (the default) counts
    what would happen and writes nothing; send ``{"dry_run": false}`` to write.
    The whole pass runs inside this request, one paged bulk at a time.
    """
    names = {c.class_id: c.class_name for c in registry.load().classes}
    return await undo_label_op(opensearch, op_id, dry_run=payload.dry_run, registry_name=names.get)
