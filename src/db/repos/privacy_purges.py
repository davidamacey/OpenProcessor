"""``privacy_purges``: one counts-only row per face or text purge.

Both calls take the caller's session and never commit. The table holds no ids, names or text:
:func:`check_counts` refuses anything but numbers and flags before the database's own ``CHECK``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.dialects.postgresql import insert

from src.db.models import PrivacyPurge


if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession


def check_counts(counts: Mapping[str, object]) -> dict[str, int | float | bool]:
    bad = [k for k, v in counts.items() if not isinstance(v, int | float | bool)]
    if bad:
        msg = f'counts must be numbers or flags; refused keys: {sorted(bad)}'
        raise ValueError(msg)
    return dict(counts)  # type: ignore[arg-type]  # narrowed by the check above


async def record(
    session: AsyncSession,
    kind: str,
    counts: Mapping[str, object],
    *,
    actor: str,
    host: str,
    at: dt.datetime | None = None,
) -> None:
    await session.execute(
        insert(PrivacyPurge).values(
            kind=kind,
            counts=check_counts(counts),
            actor=actor,
            host=host,
            **({} if at is None else {'at': at}),
        )
    )


async def import_row(
    session: AsyncSession,
    kind: str,
    counts: Mapping[str, object],
    *,
    at: dt.datetime,
    actor: str,
    host: str,
) -> bool:
    """Insert one imported record; ``False`` when a row of that kind and time already exists."""
    result = await session.execute(
        insert(PrivacyPurge)
        .values(kind=kind, counts=check_counts(counts), actor=actor, host=host, at=at)
        .on_conflict_do_nothing(index_elements=['kind', 'at'])
        .returning(PrivacyPurge.id)
    )
    return result.scalar_one_or_none() is not None
