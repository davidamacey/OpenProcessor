"""The claim log: append, the undo walk, and what the fold reads (15a sections 2.2, 3.5 and 7).

Every function takes a session already inside ``project_tx`` and never commits. The table is append
only (trigger ``claims_append_only``); nothing here updates or deletes a claim except
:mod:`src.services.erasure`, which does so under its own guard.

``append_claims`` is ``INSERT ... ON CONFLICT DO NOTHING``: with no conflict target it covers both
unique keys of the log, ``uq_claims_op_subject`` (a replayed chunk of an operation writes nothing
twice, M4) and ``uq_claims_withdraws`` (a claim is withdrawn once). "The conflict_target can be
omitted" for DO NOTHING (https://www.postgresql.org/docs/18/sql-insert.html).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import ARRAY, BigInteger, bindparam, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.claims import Claim as ClaimRow
from src.db.repos.claim_rows import CLAIM_COLUMNS, claim_from_row, draft_row
from src.domain.wire_ids import to_wire


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.domain.claims import Claim, ClaimDraft

CHUNK = 2000  # 16 columns a row: 32,000 parameters, under the protocol's 65,535

_COLUMN_LIST = ', '.join(CLAIM_COLUMNS)
_QUALIFIED = ', '.join(f'c.{c}' for c in CLAIM_COLUMNS)
_W_QUALIFIED = ', '.join(f'w.{c}' for c in CLAIM_COLUMNS)

# The live and withdrawn item and box claims of a set of items, and the withdraws naming them, in
# one statement. A claim belongs to the item whose photo and wire id it names (claim_match makes an
# item that took a claim group carry the group's id, so the id is the join). A withdraw has no
# subject of its own (it names its target), so it is found through ``uq_claims_withdraws``.
_FOR_ITEMS = text(
    f"""
    WITH wanted AS (
      SELECT item_key, photo_key, crop_id FROM items WHERE item_key = ANY(:keys)
    ), body AS (
      SELECT w.item_key, {_QUALIFIED}
      FROM wanted w JOIN claims c ON c.photo_key = w.photo_key AND c.subject_crop_id = w.crop_id
      WHERE c.subject_kind IN ('item', 'box')
    )
    SELECT item_key, {_COLUMN_LIST} FROM body
    UNION ALL
    SELECT b.item_key, {_W_QUALIFIED}
    FROM body b JOIN claims w ON w.withdraws_claim_id = b.claim_id
    ORDER BY item_key, claim_id
    """  # nosec B608  column names come from the CLAIM_COLUMNS constant, values are bound
).bindparams(bindparam('keys', type_=ARRAY(BigInteger())))

_LIVE_OF_OPERATION = text(
    """
    SELECT c.claim_id, c.photo_key, c.subject_crop_id, i.item_key
    FROM claims c
    LEFT JOIN items i ON i.photo_key = c.photo_key AND i.crop_id = c.subject_crop_id
    WHERE c.op_id = :op AND c.kind <> 'withdraw' AND c.claim_id > :after
      AND NOT EXISTS (SELECT 1 FROM claims w WHERE w.withdraws_claim_id = c.claim_id)
    ORDER BY c.claim_id
    LIMIT :n
    """
)


@dataclass(frozen=True)
class OperationClaim:
    """One live claim of an operation: what the undo walk withdraws and which item to fold."""

    claim_id: int
    photo_key: int | None
    crop_id: str | None
    item_key: int | None


async def append_claims(
    session: AsyncSession, project_id: uuid.UUID, drafts: Sequence[ClaimDraft]
) -> list[int]:
    """Append the drafts; the ids of the rows inserted, ascending. A draft whose operation key or
    withdraw target already holds a row inserts nothing (the replay of a chunk, a second undo)."""
    inserted: list[int] = []
    for start in range(0, len(drafts), CHUNK):
        rows = [draft_row(project_id, d) for d in drafts[start : start + CHUNK]]
        result = await session.execute(
            pg_insert(ClaimRow).values(rows).on_conflict_do_nothing().returning(ClaimRow.claim_id)
        )
        inserted.extend(r.claim_id for r in result)
    return sorted(inserted)


async def load_claims_for_items(
    session: AsyncSession, item_keys: Iterable[int]
) -> dict[int, list[Claim]]:
    """The claims of each item, withdraws included, in claim order: ``ItemInputs.claims``."""
    wanted = sorted(set(item_keys))
    claims: dict[int, list[Claim]] = {k: [] for k in wanted}
    if not wanted:
        return claims
    rows = await session.execute(_FOR_ITEMS, {'keys': wanted})
    for row in rows:
        claims[row.item_key].append(claim_from_row(row))
    return claims


async def live_claims_of_operation(
    session: AsyncSession, op_id: str, *, after_claim_id: int = 0, limit: int = 1000
) -> list[OperationClaim]:
    """The next page of an operation's claims no withdraw names yet, by claim id: the undo walk
    (15a section 3.5). A dead job resumes with ``after_claim_id`` 0 and finds only what is left."""
    rows = await session.execute(
        _LIVE_OF_OPERATION, {'op': uuid.UUID(op_id), 'after': after_claim_id, 'n': limit}
    )
    return [
        OperationClaim(
            r.claim_id,
            r.photo_key,
            None if r.subject_crop_id is None else to_wire(r.subject_crop_id),
            r.item_key,
        )
        for r in rows
    ]
