"""Item identity rows, the item lock and the sequences machine writers keep (15a sections 2.3 and 7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.items import Item
from src.db.repos.upserts import any_of
from src.domain.wire_ids import to_uuid, to_wire


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

CHUNK = 4000


@dataclass(frozen=True)
class NewItem:
    photo_key: int
    crop_id: str


async def insert_items(
    session: AsyncSession, project_id: uuid.UUID, items: Sequence[NewItem]
) -> dict[str, int]:
    """The wire id to ``item_key`` of the items stored now; an item that exists is left alone."""
    stored: dict[str, int] = {}
    for start in range(0, len(items), CHUNK):
        result = await session.execute(
            pg_insert(Item)
            .values(
                [
                    {
                        'project_id': project_id,
                        'photo_key': i.photo_key,
                        'crop_id': to_uuid(i.crop_id),
                    }
                    for i in items[start : start + CHUNK]
                ]
            )
            .on_conflict_do_nothing(index_elements=['project_id', 'crop_id'])
            .returning(Item.item_key, Item.crop_id)
        )
        stored.update({to_wire(r.crop_id): r.item_key for r in result})
    return stored


async def item_keys_by_crop_id(
    session: AsyncSession, project_id: uuid.UUID, crop_ids: Iterable[str]
) -> dict[str, int]:
    wanted = sorted({to_uuid(c) for c in crop_ids})
    if not wanted:
        return {}
    rows = await session.execute(
        select(Item.crop_id, Item.item_key)
        .where(Item.project_id == project_id, any_of(Item.crop_id, wanted))
        .order_by(Item.item_key)
    )
    return {to_wire(r.crop_id): r.item_key for r in rows}


async def item_keys_of_photos(session: AsyncSession, photo_keys: Iterable[int]) -> list[int]:
    """Every item of the photos, in key order (no lock)."""
    wanted = sorted(set(photo_keys))
    if not wanted:
        return []
    rows = await session.execute(
        select(Item.item_key).where(any_of(Item.photo_key, wanted)).order_by(Item.item_key)
    )
    return [r.item_key for r in rows]


async def photo_keys_of_items(session: AsyncSession, item_keys: Iterable[int]) -> list[int]:
    wanted = sorted(set(item_keys))
    if not wanted:
        return []
    rows = await session.execute(
        select(Item.photo_key)
        .where(any_of(Item.item_key, wanted))
        .distinct()
        .order_by(Item.photo_key)
    )
    return [r.photo_key for r in rows]


async def lock_items(session: AsyncSession, item_keys: Iterable[int]) -> list[int]:
    """``SELECT ... WHERE item_key = ANY(...) ORDER BY item_key FOR UPDATE`` (rank 6 of the lock
    order, 15a section 7): the keys that exist, locked in ascending order, whatever order the
    caller named them in, so two writers over the same items wait for each other and never cycle
    ("acquire locks on multiple objects in a consistent order",
    https://www.postgresql.org/docs/18/explicit-locking.html, "Deadlocks"). ``item_key`` never
    changes, so concurrent updates cannot reorder the rows the ``ORDER BY`` sorted
    (https://www.postgresql.org/docs/18/sql-select.html#SQL-FOR-UPDATE-SHARE, the caveat on ORDER BY).
    A requested key that is not returned was deleted meanwhile: the caller skips it or answers 404.
    """
    wanted = sorted(set(item_keys))
    if not wanted:
        return []
    rows = await session.execute(
        select(Item.item_key)
        .where(any_of(Item.item_key, wanted))
        .order_by(Item.item_key)
        .with_for_update()
    )
    return [r.item_key for r in rows]
