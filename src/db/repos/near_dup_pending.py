"""The ``near_dup_pending`` queue: mark members stale, claim them for one job, drain or expire them.

Split from ``near_dup.py`` by concern (the queue vs the run, edge and group tables). Every function
takes a session already inside the scope that owns the rows and never commits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.near_dup import NearDupPending


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

CHUNK = 4000


def same_project(column: Any, project_id: uuid.UUID | None) -> Any:
    return column.is_(None) if project_id is None else column == project_id


# 4 bound values per row against the 32767 parameter limit of the wire protocol.
PENDING_CHUNK = 2000


async def enqueue_pending(
    session: AsyncSession,
    project_id: uuid.UUID | None,
    level: str,
    member_ids: Iterable[str],
    reason: str,
) -> int:
    """Mark members stale; returns how many distinct members were written.

    A member that is already pending keeps its single row: the newest reason wins, the row is
    re-aged and released from any job that claimed it, so a change that lands while a rescore
    runs is never lost when that rescore drains its claim.
    """
    ids = sorted(set(member_ids))
    for start in range(0, len(ids), PENDING_CHUNK):
        chunk = ids[start : start + PENDING_CHUNK]
        insert = pg_insert(NearDupPending).values(
            [
                {'project_id': project_id, 'level': level, 'member_id': member, 'reason': reason}
                for member in chunk
            ]
        )
        await session.execute(
            insert.on_conflict_do_update(
                constraint='uq_near_dup_pending_member',
                set_={
                    'reason': insert.excluded.reason,
                    'enqueued_at': func.now(),
                    'claimed_by_job_id': None,
                },
            )
        )
    return len(ids)


@dataclass(frozen=True)
class ClaimedMember:
    member_id: str
    reason: str


async def claim_pending(
    session: AsyncSession, project_id: uuid.UUID | None, level: str, job_id: uuid.UUID, limit: int
) -> list[ClaimedMember]:
    """Claim the oldest pending rows of one (project, level) for ``job_id`` with ``SKIP LOCKED``.

    Only one ``near_dup`` job per (project, level) is active (the exclusive group), so a row a
    dead predecessor claimed is safe to take over: every pending row of the level is claimable.
    """
    oldest = (
        select(NearDupPending.id)
        .where(same_project(NearDupPending.project_id, project_id), NearDupPending.level == level)
        .order_by(NearDupPending.enqueued_at, NearDupPending.id)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    rows = await session.execute(
        update(NearDupPending)
        .where(NearDupPending.id.in_(oldest))
        .values(claimed_by_job_id=job_id)
        .returning(NearDupPending.member_id, NearDupPending.reason)
    )
    return [ClaimedMember(r.member_id, r.reason) for r in rows]


async def drain_claimed(session: AsyncSession, job_id: uuid.UUID, member_ids: Iterable[str]) -> int:
    """Delete the rows ``job_id`` still holds for ``member_ids``; a row re-enqueued meanwhile was
    released from the claim by :func:`enqueue_pending` and stays."""
    ids = list(member_ids)
    done = 0
    for start in range(0, len(ids), CHUNK):
        result = await session.execute(
            delete(NearDupPending).where(
                NearDupPending.claimed_by_job_id == job_id,
                NearDupPending.member_id.in_(ids[start : start + CHUNK]),
            )
        )
        done += result.rowcount  # type: ignore[attr-defined]
    return done


async def expire_claimed(
    session: AsyncSession, job_id: uuid.UUID, member_ids: Iterable[str], max_age: dt.timedelta
) -> list[str]:
    """Delete the rows ``job_id`` holds for ``member_ids`` that were enqueued more than
    ``max_age`` ago; returns their member ids (a re-enqueue re-ages a row, so only a member
    that stayed unresolved the whole time expires)."""
    ids = list(member_ids)
    expired: list[str] = []
    for start in range(0, len(ids), CHUNK):
        rows = await session.execute(
            delete(NearDupPending)
            .where(
                NearDupPending.claimed_by_job_id == job_id,
                NearDupPending.member_id.in_(ids[start : start + CHUNK]),
                NearDupPending.enqueued_at < func.now() - max_age,
            )
            .returning(NearDupPending.member_id)
        )
        expired.extend(r.member_id for r in rows)
    return sorted(expired)
