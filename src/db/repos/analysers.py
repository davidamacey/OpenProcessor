"""Analysers and the analysis ledger (15a section 2.3, R4).

An analyser row is immutable and shared by every project (``analysers`` has no project column): a new
prompt-pack revision or model build is a new row, and a configuration that returns to an earlier
state reuses the earlier row, so the re-run work list is an equality ("the photo's analyser is not
the project's current one, or its state is not ``done``"), never a "below" (15a m2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.analysis import Analyser, AnalysisLedger
from src.db.repos.upserts import upsert_distinct


if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


def config_sha256(params: Mapping[str, Any]) -> str:
    """The hash of an analyser's parameters: canonical JSON (sorted keys, no spaces), so the same
    configuration always names the same row."""
    canonical = json.dumps(params, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(canonical.encode('ascii')).hexdigest()


async def get_or_create_analyser(
    session: AsyncSession,
    *,
    kind: str,
    name: str,
    version: str,
    params: Mapping[str, Any],
    config_ref: Mapping[str, Any] | None = None,
) -> int:
    """The ``analyser_id`` of (kind, name, version, configuration), created when absent.

    ``ON CONFLICT DO NOTHING`` waits for a concurrent transaction that inserted the same identity
    to commit or abort, so the select that follows sees its row
    (https://www.postgresql.org/docs/18/sql-insert.html, "ON CONFLICT Clause"; READ COMMITTED takes
    a new snapshot per statement).
    """
    digest = config_sha256(params)
    values = {
        'kind': kind,
        'name': name,
        'version': version,
        'config_sha256': digest,
        'config_ref': dict(config_ref or {}),
    }
    inserted = await session.execute(
        pg_insert(Analyser)
        .values(values)
        .on_conflict_do_nothing(constraint='uq_analysers_identity')
        .returning(Analyser.analyser_id)
    )
    found = inserted.scalar_one_or_none()
    if found is not None:
        return int(found)
    return int(
        (
            await session.execute(
                select(Analyser.analyser_id).where(
                    Analyser.kind == kind,
                    Analyser.name == name,
                    Analyser.version == version,
                    Analyser.config_sha256 == digest,
                )
            )
        ).scalar_one()
    )


@dataclass(frozen=True)
class LedgerEntry:
    photo_key: int
    kind: str
    analyser_id: int
    state: str
    reason: str | None = None


async def record_ledger(
    session: AsyncSession, project_id: uuid.UUID, entries: Sequence[LedgerEntry]
) -> int:
    """Set what has been computed for each photo and analyser kind; an entry that says what the
    ledger already says writes nothing. ``updated_at`` moves only with another column."""
    return await upsert_distinct(
        session,
        AnalysisLedger,
        [
            {
                'photo_key': e.photo_key,
                'project_id': project_id,
                'kind': e.kind,
                'analyser_id': e.analyser_id,
                'state': e.state,
                'reason': e.reason,
            }
            for e in entries
        ],
        key=('photo_key', 'kind'),
        set_extra={'updated_at': func.now()},
    )


_WORK_LIST = text(
    """
    SELECT p.photo_key
    FROM photos p
    LEFT JOIN analysis_ledger l ON l.photo_key = p.photo_key AND l.kind = :kind
    WHERE p.project_id = :project AND p.photo_key > :after
      AND (l.photo_key IS NULL OR l.analyser_id <> :analyser OR l.state <> 'done')
    ORDER BY p.photo_key
    LIMIT :n
    """
)


async def photos_needing(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    kind: str,
    analyser_id: int,
    after_photo_key: int = 0,
    limit: int = 1000,
) -> list[int]:
    """The next page of the re-run work list of one analyser kind: photos with no ledger row, a row
    of another analyser, or a row that is not ``done``."""
    rows = await session.execute(
        _WORK_LIST,
        {
            'kind': kind,
            'project': project_id,
            'after': after_photo_key,
            'analyser': analyser_id,
            'n': limit,
        },
    )
    return [r.photo_key for r in rows]
