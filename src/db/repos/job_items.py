"""``job_items``: the explicit item list of a job (reprocess targets, assist-run members).

Written in the start transaction through ``start_job(with_rows=...)``, read by the handler in
pages keyed on ``position`` so a resumed attempt continues where the cursor says. Every function
takes a caller-owned session and never commits.
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING

from sqlalchemy import insert, select

from src.db.models import JobItem


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

INSERT_BATCH = 5_000
"""Rows per INSERT: three bind values each stays far below the 65,535-parameter limit."""


async def add_items(session: AsyncSession, job_id: uuid.UUID, item_ids: Iterable[str]) -> int:
    """Store ``item_ids`` in order (positions from 0); returns how many were stored."""
    iterator = iter(item_ids)
    position = 0
    while batch := list(islice(iterator, INSERT_BATCH)):
        await session.execute(
            insert(JobItem),
            [
                {'job_id': job_id, 'position': position + offset, 'item_id': item_id}
                for offset, item_id in enumerate(batch)
            ],
        )
        position += len(batch)
    return position


async def read_items(
    session: AsyncSession, job_id: uuid.UUID, *, start: int = 0, limit: int
) -> list[str]:
    """The ids at positions ``start`` .. ``start + limit - 1``, in position order."""
    rows = await session.execute(
        select(JobItem.item_id)
        .where(JobItem.job_id == job_id, JobItem.position >= start)
        .order_by(JobItem.position)
        .limit(limit)
    )
    return list(rows.scalars())
