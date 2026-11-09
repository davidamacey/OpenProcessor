"""Rows the holdout freeze writes for near-duplicate moves.

Takes a session already inside ``project_tx`` and never commits: the rows belong to the freeze's
own transaction, so a database error refuses the freeze.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.datasets import HoldoutFreeze
from src.db.models.near_dup import HoldoutNearDupMove


if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

# 6 bound values per row against the 32767 parameter limit of the wire protocol.
MOVE_CHUNK = 2000


@dataclass(frozen=True)
class Move:
    item_id: str
    anchor_item_id: str
    group_id: uuid.UUID


async def record_moves(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    freeze_id: uuid.UUID,
    run_id: uuid.UUID | None,
    moves: Sequence[Move],
) -> None:
    """One row per moved item. A rerun of the freeze (``force``) rewrites the row of an item
    already moved, so the row always names the latest freeze."""
    for start in range(0, len(moves), MOVE_CHUNK):
        insert = pg_insert(HoldoutNearDupMove).values(
            [
                {
                    'project_id': project_id,
                    'item_id': m.item_id,
                    'anchor_item_id': m.anchor_item_id,
                    'group_id': m.group_id,
                    'freeze_id': freeze_id,
                    'run_id': run_id,
                }
                for m in moves[start : start + MOVE_CHUNK]
            ]
        )
        await session.execute(
            insert.on_conflict_do_update(
                constraint='uq_holdout_near_dup_moves_item',
                set_={
                    'anchor_item_id': insert.excluded.anchor_item_id,
                    'group_id': insert.excluded.group_id,
                    'freeze_id': insert.excluded.freeze_id,
                    'run_id': insert.excluded.run_id,
                },
            )
        )


async def stamp_freeze(
    session: AsyncSession,
    freeze_id: uuid.UUID,
    *,
    image_run_id: uuid.UUID | None,
    item_run_id: uuid.UUID | None,
    moved: int,
) -> None:
    await session.execute(
        update(HoldoutFreeze)
        .where(HoldoutFreeze.id == freeze_id)
        .values(
            near_dup_image_run_id=image_run_id,
            near_dup_item_run_id=item_run_id,
            moved_by_near_dup=moved,
        )
    )
