"""Reads the training export makes of the current run's groups (no writes, no commits)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from src.db.models.near_dup import NearDupMember
from src.domain.near_dup import Seat


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

MEMBER_CHUNK = 4000


async def dedup_seats(
    session: AsyncSession, run_id: uuid.UUID, member_ids: Iterable[str]
) -> dict[str, Seat]:
    """The dedup-group seat of each of ``member_ids`` that is grouped in ``run_id``."""
    ids = sorted(set(member_ids))
    seats: dict[str, Seat] = {}
    for start in range(0, len(ids), MEMBER_CHUNK):
        rows = await session.execute(
            select(
                NearDupMember.member_id,
                NearDupMember.group_id,
                NearDupMember.is_representative,
                NearDupMember.cosine_to_rep,
            ).where(
                NearDupMember.run_id == run_id,
                NearDupMember.purpose == 'dedup',
                NearDupMember.member_id.in_(ids[start : start + MEMBER_CHUNK]),
            )
        )
        for r in rows:
            seats[r.member_id] = Seat(str(r.group_id), r.is_representative, float(r.cosine_to_rep))
    return seats
