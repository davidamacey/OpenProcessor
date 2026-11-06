"""Near-duplicate repository: pending rows, candidate edges, runs, groups and decisions.

Every function takes a session already inside ``project_tx`` (or ``deployment_tx`` for the core
index) and never commits. A pending row is written before the OpenSearch write it describes:
an extra row is harmless because the rescore is idempotent per member, a missing one would be a
silent stale group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.near_dup import NearDupPending


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

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
