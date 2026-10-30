"""Curation router sub-module — recorded bulk label and move operations."""

from __future__ import annotations

from typing import Any

from fastapi import Query

from src.routers.curation._common import OpenSearchDep, router
from src.schemas.curation.label_ops import LabelOpList
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
