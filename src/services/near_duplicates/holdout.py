"""The holdout guard: a held-out item's near-duplicates are held out with it.

At freeze, ``H`` is the selected cohort ``H0`` plus every member of every current item-level
``leak`` group that holds an item of ``H0`` (a group is a connected component, so the closure is
transitive by construction). The flag goes through the OCC bulk writer; a conflict or an error
refuses the freeze, so the record always covers exactly the flagged ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.clients.occ import occ_skip_on_conflict_bulk
from src.core.exceptions import RequestRejectedError
from src.db.repos import near_dup as repo
from src.db.repos.near_dup_holdout import Move, record_moves, stamp_freeze
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    import uuid
    from collections.abc import Collection

    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class Closure:
    moves: tuple[Move, ...]
    image_run_id: uuid.UUID | None
    item_run_id: uuid.UUID | None

    @property
    def moved_ids(self) -> list[str]:
        return [m.item_id for m in self.moves]


async def leak_closure(
    session: AsyncSession, project_id: uuid.UUID, selected: Collection[str]
) -> Closure:
    """The items the current item-level leak groups pull into the holdout with ``selected``.

    Runs inside ``project_tx``. Each moved item names its anchor (the lowest-id selected member
    of its group). A project with no current item run moves nothing.
    """
    image = await repo.current_run(session, project_id, 'image')
    item = await repo.current_run(session, project_id, 'item')
    image_id = None if image is None else image.id
    if item is None:
        return Closure((), image_id, None)
    held = set(selected)
    moves: dict[str, Move] = {}
    for group in await repo.load_groups(session, item.id, 'leak'):
        anchors = sorted(held.intersection(group.members))
        if not anchors:
            continue
        for member in group.members:
            if member not in held:
                moves.setdefault(member, Move(member, anchors[0], group.lineage_id))
    return Closure(tuple(sorted(moves.values(), key=lambda m: m.item_id)), image_id, item.id)


async def flag_held_out(client: AsyncOpenSearch, index: str, item_ids: list[str]) -> None:
    """Set ``test_holdout`` on every id, or raise 502 when any write did not land."""
    now = now_iso()
    result = await occ_skip_on_conflict_bulk(
        client,
        doc_ids=item_ids,
        merger=lambda _id, _source: {'test_holdout': True, 'updated_at': now},
        index=index,
        refresh=True,
        writer_id='holdout_freeze',
    )
    unflagged = result['skipped_ids'] + [e['doc_id'] for e in result['errors']]
    if unflagged:
        raise RequestRejectedError(
            status_code=502,
            detail=(
                f'holdout freeze incomplete: {len(unflagged)} of {len(item_ids)} items '
                f'were not flagged (ids: {unflagged[:20]}); no record persisted, retry'
            ),
        )


async def record_closure(
    session: AsyncSession, project_id: uuid.UUID, freeze_id: uuid.UUID, closure: Closure
) -> None:
    """Write the move rows and the run ids and count on the freeze record (same transaction)."""
    await record_moves(
        session, project_id, freeze_id=freeze_id, run_id=closure.item_run_id, moves=closure.moves
    )
    await stamp_freeze(
        session,
        freeze_id,
        image_run_id=closure.image_run_id,
        item_run_id=closure.item_run_id,
        moved=len(closure.moves),
    )
