"""Mark members stale for the near-duplicate rescore, before the write that changed them.

``enqueue_pending(session, project, level, member_ids, reason)`` is the one entry point every
writer calls (curation ingest, re-embed, deletes, dataset import and combine). The call goes
before the OpenSearch write it describes: if the write then fails, the extra row costs one
idempotent rescore; if the row were written after, a crash between the two would leave a stale
group nobody knows about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.curation_roles import IndexRole
from src.config.project_context import current_project
from src.db.repos import near_dup as repo
from src.db.session import async_session_factory, project_tx
from src.domain import near_dup as rules
from src.services.config_store.bound_project import bound_project_id


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession


class PendingError(ValueError):
    """The level or the reason is not one the pending table holds."""


async def enqueue_pending(
    session: AsyncSession,
    project: uuid.UUID | None,
    level: str,
    member_ids: Iterable[str],
    reason: str,
) -> int:
    """Insert (or refresh) one pending row per member; returns the number of members.

    ``project`` is the project id, or ``None`` for the core global index. Both ``level`` and
    ``reason`` are checked here so a typo is an error at the call site, not a CHECK violation
    in a later flush.
    """
    if level not in rules.LEVELS:
        msg = f'unknown near-duplicate level {level!r}'
        raise PendingError(msg)
    if reason not in rules.PENDING_REASONS:
        msg = f'unknown pending reason {reason!r}'
        raise PendingError(msg)
    return await repo.enqueue_pending(session, project, level, member_ids, reason)


async def enqueue_pending_bound(level: str, member_ids: Iterable[str], reason: str) -> int:
    """:func:`enqueue_pending` in its own transaction, for the bound project."""
    ids = list(member_ids)
    if not ids:
        return 0
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        return await enqueue_pending(session, project_id, level, ids, reason)


def level_of_index(index: str) -> str:
    """The near-duplicate level of one of the bound project's index names (fail closed)."""
    indexes = current_project().record.resources.indexes
    if indexes.get(IndexRole.IMAGES) == index:
        return 'image'
    if indexes.get(IndexRole.ITEMS) == index:
        return 'item'
    msg = f'index {index!r} is not an images or items index of the bound project'
    raise PendingError(msg)
