"""Served pages over the stored result of a finished ``selection`` job.

The job keeps its picked ids in ``result.selection.crop_ids``; the Workspace reads them through
``GET /select/diverse/{job_id}/items`` a page at a time instead of receiving the whole id list.
The cursor is the shared :mod:`~src.services.curation.list_cursor` one carrying the offset into the
stored list, so a cursor of another list is refused the same way.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from src.config.project_context import current_project_id
from src.services.curation.list_cursor import CursorError, decode_cursor, encode_cursor
from src.services.curation.selection.job import KIND
from src.services.jobs import status
from src.services.jobs.service import rejection


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

SORT_ID = 'select_diverse'


async def stored_selection(session: AsyncSession, job_id: uuid.UUID) -> tuple[str, dict[str, Any]]:
    """``(status, selection)`` of the project's own ``selection`` job; 404 for any other job.

    ``get_job`` is project-scoped, so another project's job id is a 404 like an unknown one."""
    view = await status.get_job(session, job_id, project=current_project_id())
    if view.kind != KIND:
        raise rejection(404, 'job_not_found', f'no selection job {job_id}')
    return view.status, dict((view.result or {}).get('selection') or {})


def page_slice(ids: list[str], *, cursor: str | None, limit: int) -> tuple[list[str], str | None]:
    """The ids of one page and the cursor of the next, or ``None`` on the last page.

    Raises :class:`CursorError` for a cursor of another sort or one that is not an offset."""
    start = 0
    if cursor is not None:
        after = decode_cursor(cursor, sort_id=SORT_ID)
        if len(after) != 1 or not isinstance(after[0], int) or after[0] < 0:
            msg = 'cursor is not one this route served'
            raise CursorError(msg)
        start = after[0]
    end = start + limit
    return ids[start:end], (encode_cursor(SORT_ID, [end]) if end < len(ids) else None)
