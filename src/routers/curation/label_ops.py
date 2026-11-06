"""Curation router sub-module — recorded bulk label and move operations."""

from __future__ import annotations

from typing import Any

from fastapi import Query
from fastapi.responses import JSONResponse

from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers.curation._common import OpenSearchDep, RegistryDep, router
from src.routers.curation._selection_write_models import LabelOpAccepted
from src.schemas.curation.label_ops import LabelOpList, LabelOpUndoRequest, LabelOpUndoResponse
from src.services.curation.label_ops.undo import start as undo_start, undo_dry_run
from src.services.curation.label_ops.views import list_label_ops


@router.get('/label_ops', response_model=LabelOpList)
async def get_label_ops(
    opensearch: OpenSearchDep,
    limit: int = Query(default=20, ge=1, le=50),
    undoable: bool | None = Query(default=None, description='true: only undoable operations.'),
    after: str | None = Query(default=None, description='`next_after` of the previous page.'),
) -> dict[str, Any]:
    """The project's recorded bulk label and move operations, newest first.

    ``current`` is how many items still carry the operation's label; an
    operation is ``undoable`` while that is above zero and it is not being
    written or undone. Class ids are never served: identity is the name.
    Page with ``after`` (the previous page's ``next_after``); ``next_after`` is
    null at the end. A malformed ``after`` is 422 ``invalid_request``.
    """
    return await list_label_ops(opensearch, limit=limit, undoable_only=bool(undoable), after=after)


@router.post(
    '/label_ops/{op_id}/undo',
    response_model=None,
    responses={
        200: {'model': LabelOpUndoResponse, 'description': 'A dry run: counts, nothing written.'},
        202: {'model': LabelOpAccepted, 'description': 'The undo pass is queued as a job.'},
        404: {'description': 'label_op_not_found: unknown, malformed or pruned operation.'},
        409: {
            'description': 'already_undone, job_busy (another undo is running) or '
            'label_op_in_progress (the operation is still being written).'
        },
    },
)
async def undo_op(
    op_id: str,
    payload: LabelOpUndoRequest,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
    session: DbSessionDep,
) -> Any:
    """Undo a bulk label or move, item by item, from the stored operation.

    An item is restored only while it still carries the operation's label and
    nothing changed it since: a later edit, an exclusion, a class merge or the
    test holdout keeps the item as it is, and a restore target whose class was
    renamed is refused (``kept_stale_class``). ``dry_run`` (the default) counts
    what would happen and writes nothing; send ``{"dry_run": false}`` to queue
    the pass as a ``label_undo`` job (202 with the job id and the ``op_id``).
    """
    if payload.dry_run:
        names = {c.class_id: c.class_name for c in (await registry.snapshot()).classes}
        return await undo_dry_run(opensearch, op_id, registry_name=names.get)
    job = await undo_start(session, current_project_id(), op_id, opensearch)
    return JSONResponse(
        status_code=202, content=LabelOpAccepted(op_id=op_id, job_id=str(job.id)).model_dump()
    )
