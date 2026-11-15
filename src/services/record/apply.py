"""The heart of every writer of the record: append, upsert, load, fold, store (15a sections 4.1 and 7).

``apply`` runs inside the caller's transaction (``project_tx``) and never commits. It takes the item
locks in the plan's order, appends the claims, upserts the machine results, loads everything the
fold needs for the locked items, folds them with :func:`src.domain.fold.fold_item`, and writes the
state rows that differ. The state of an item after ``apply`` is therefore always the fold of the
rows the transaction can see: a second writer over the same items waits on the item lock and folds the
first one's committed claims and results, so no claim is lost and no state row differs from a fresh fold.

No route or writer calls it yet: the lanes of storage steps 2 to 4 do.

Lock order (15a section 7): rank 6 is ``items``, ``FOR UPDATE`` in ascending ``item_key``. Callers that
need a lower rank (the operation row, the registry row, the imohash locks, the photo) take it before
they call ``apply``; ``apply`` takes none of those. Because the rank of an item among its photo's
items depends on every sibling's box, ``apply`` locks and folds all items of the touched photos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.db.repos.claims import append_claims
from src.db.repos.fold_load import TextConsensus, load_item_inputs, load_registry
from src.db.repos.items import item_keys_of_photos, lock_items, photo_keys_of_items
from src.db.repos.results import ResultWrite, upsert_results
from src.db.repos.state import delete_states, upsert_box_states, upsert_item_states
from src.domain.fold import fold_item, rank_items


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.domain.claims import ClaimDraft
    from src.domain.fold_state import Conflict, FoldSettings, ItemFold


@dataclass(frozen=True)
class FoldContext:
    """What the fold needs that is neither a row nor a claim: deployment values and the reading
    consensus (see :data:`src.db.repos.fold_load.TextConsensus`)."""

    settings: FoldSettings
    text_consensus: TextConsensus


@dataclass(frozen=True)
class ApplyOutcome:
    appended: tuple[int, ...]
    """Ids of the claims inserted now (a replay of an operation chunk appends none)."""
    locked: tuple[int, ...]
    """The items locked and folded: the requested ones that exist and their photos' siblings."""
    missing: tuple[int, ...]
    """Requested items that no longer exist: deleted while the caller decided (skip, or 404)."""
    vanished: tuple[int, ...]
    """Folded items with neither a detection nor a live asserting claim: their state rows are gone
    and the attach step deletes their identity rows."""
    state_rows_written: int
    conflicts: tuple[Conflict, ...] = field(default=())
    """Import or combine claims that lost to a person's (``locked_conflict``, ``combine_conflict``)."""


async def apply(
    session: AsyncSession,
    project: uuid.UUID,
    *,
    claims: Sequence[ClaimDraft] = (),
    results: Sequence[ResultWrite] = (),
    item_keys: Iterable[int],
    context: FoldContext,
) -> ApplyOutcome:
    requested = sorted(set(item_keys))
    photos = await photo_keys_of_items(session, requested)
    siblings = await item_keys_of_photos(session, photos)
    locked = await lock_items(session, siblings)
    missing = tuple(k for k in requested if k not in set(locked))

    appended = await append_claims(session, project, claims) if claims else []
    if results:
        await upsert_results(session, project, results)

    loaded = await load_item_inputs(session, locked, text_consensus=context.text_consensus)
    registry = await load_registry(session, project)
    folds: dict[int, ItemFold] = {}
    vanished: list[int] = []
    for key in locked:
        fold = fold_item(loaded[key].inputs, registry, context.settings)
        if fold is None:
            vanished.append(key)
        else:
            folds[key] = fold
    folds = _ranked(folds)

    written = await upsert_item_states(session, project, folds)
    written += await upsert_box_states(
        session, project, folds, {k: loaded[k].box_keys for k in folds}
    )
    await delete_states(session, vanished)
    conflicts = tuple(c for k in sorted(folds) for c in folds[k].conflicts)
    return ApplyOutcome(
        tuple(appended), tuple(locked), missing, tuple(vanished), written, conflicts
    )


def _ranked(folds: dict[int, ItemFold]) -> dict[int, ItemFold]:
    """Rank each photo's items by box area (``rank_items``): the folds of one photo together."""
    by_photo: dict[int, list[int]] = {}
    for key, fold in folds.items():
        by_photo.setdefault(fold.state.photo_key, []).append(key)
    return {
        key: fold
        for keys in by_photo.values()
        for key, fold in zip(keys, rank_items([folds[k] for k in keys]), strict=True)
    }
