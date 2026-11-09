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
from src.db.repos.records import current_holdout_freeze
from src.db.session import project_tx
from src.services.embedding_spaces.readers import vector_source_excludes
from src.services.events.publish import apublish
from src.services.near_duplicates import status as near_dup_status
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
        vector_excludes=await vector_source_excludes(),
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


async def _stale_levels(session: AsyncSession, project_id: uuid.UUID) -> list[dict[str, object]]:
    current = await near_dup_status.status(session, project_id)
    return [
        {
            'level': lv.level,
            'stale_reasons': list(lv.stale_reasons),
            'pending_count': lv.pending_count,
        }
        for lv in current.levels
        if lv.stale_reasons
    ]


def _refuse(levels: list[dict[str, object]]) -> RequestRejectedError:
    return RequestRejectedError(
        status_code=409, detail={'code': 'near_dup_stale', 'levels': levels}
    )


async def _has_freeze(session: AsyncSession, project_id: uuid.UUID) -> bool:
    async with project_tx(session, project_id):
        return await current_holdout_freeze(session, project_id) is not None


async def require_fresh_when_frozen(session: AsyncSession, project_id: uuid.UUID) -> None:
    """Refuse (409 ``near_dup_stale``) a training export while the groups are stale, but only
    once a holdout is frozen (owner decision 2026-11-08): before any freeze there is no holdout
    for a near-duplicate to leak across, so an export is never refused for this. The body names
    each stale level so the status route can say which job to start."""
    if not await _has_freeze(session, project_id):
        return
    stale = await _stale_levels(session, project_id)
    if stale:
        raise _refuse(stale)


async def require_freezable(session: AsyncSession, project_id: uuid.UUID) -> None:
    """Refuse (409 ``near_dup_stale``) a freeze that would split near-duplicates across it.

    A re-freeze is refused while any level is stale. The first freeze pulls the near-duplicates
    of its cohort in from the current item-level run, so it is refused only when there is no
    current item-level run at all.
    """
    if await _has_freeze(session, project_id):
        stale = await _stale_levels(session, project_id)
    else:
        stale = [
            lv
            for lv in await _stale_levels(session, project_id)
            if lv['level'] == 'item' and lv['stale_reasons'] == ['no_run']
        ]
    if stale:
        raise _refuse(stale)


MGET_CHUNK = 500


async def _held_out(client: AsyncOpenSearch, index: str, ids: list[str]) -> set[str]:
    held: set[str] = set()
    for start in range(0, len(ids), MGET_CHUNK):
        resp = await client.mget(
            index=index,
            body={'ids': ids[start : start + MGET_CHUNK]},
            _source_includes=['test_holdout'],
        )
        held.update(
            d['_id'] for d in resp['docs'] if d.get('found') and d['_source'].get('test_holdout')
        )
    return held


async def absorb(
    session: AsyncSession,
    client: AsyncOpenSearch,
    *,
    index: str,
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    leak: Collection[repo.GroupToWrite],
) -> int:
    """After a freeze: move the non-held-out members of mixed leak groups into the holdout.

    Called by an item-level run before it flips current, inside its transaction. A project with no
    frozen holdout absorbs nothing. The flag is set first (OCC writer, a failed write fails the
    run), then the move rows and the ``holdout.near_dup_absorbed`` event are written with the
    run's own transaction. Returns the number of items moved.
    """
    freeze = await current_holdout_freeze(session, project_id)
    if freeze is None:
        return 0
    grouped = [g for g in leak if len(g.members) > 1]
    held = await _held_out(client, index, sorted({m for g in grouped for m in g.members}))
    moves: dict[str, Move] = {}
    for group in grouped:
        anchors = sorted(held.intersection(group.members))
        if not anchors:
            continue
        for member in group.members:
            if member not in held:
                moves.setdefault(member, Move(member, anchors[0], group.lineage_id))
    if not moves:
        return 0
    await flag_held_out(client, index, sorted(moves))
    await record_moves(
        session, project_id, freeze_id=freeze.id, run_id=run_id,
        moves=sorted(moves.values(), key=lambda m: m.item_id),
    )  # fmt: skip
    await apublish(
        'holdout.near_dup_absorbed',
        {'run_id': str(run_id), 'moved': len(moves), 'freeze_id': str(freeze.id)},
        project=project_id,
        session=session,
    )
    return len(moves)
