"""The exact-content key: one ``image_contents`` row per stored image.

The row carries the sampled ``imohash`` of the image's bytes (always) and its SHA-256 (only once a
second file with that ``imohash`` made the full comparison necessary). ``imohash`` is not unique:
two different files can share it, and the SHA-256 tells them apart (UNIQUE per project where it is
set).

Every function takes a session already inside ``project_tx`` (``core_tx`` for the core global
index, whose rows have no project: ``project_id`` ``None``) and never commits. A lookup and the
insert that follows it must run in one transaction under :func:`lock_imohash`, so concurrent
callers with the same content are decided one after the other, across every process (#358).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.near_dup import ImageContent


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

CHUNK = 4000


def _owner(project_id: uuid.UUID | None) -> Any:
    """The project filter: the project's rows, or the core rows (no project)."""
    return (
        ImageContent.project_id.is_(None)
        if project_id is None
        else (ImageContent.project_id == project_id)
    )


class ContentKeyConflictError(ValueError):
    """The image already has a content row for other bytes (its file changed under it)."""


@dataclass
class ContentRow:
    """One stored image's content row."""

    id: int
    image_id: str
    image_path: str
    size_bytes: int
    content_sha256: str | None


def _lock_key(project_id: uuid.UUID | None, imohash: str) -> str:
    return f'image_contents:{project_id}:{imohash}'


async def lock_imohash(session: AsyncSession, project_id: uuid.UUID | None, imohash: str) -> None:
    """``pg_advisory_xact_lock`` on a 64-bit key of (project, imohash), held to the end of the
    transaction. Two imohashes that share a hashed key only wait for each other."""
    await session.execute(
        text('SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))'),
        {'key': _lock_key(project_id, imohash)},
    )


async def lock_imohashes(
    session: AsyncSession, project_id: uuid.UUID | None, imohashes: Sequence[str]
) -> None:
    """:func:`lock_imohash` for every ``imohash``, taken one after the other in the order given
    (callers pass them sorted, so two transactions that take several locks never wait on each
    other in a cycle: https://www.postgresql.org/docs/18/explicit-locking.html, "Deadlocks").

    One statement: the ordered subquery with ``OFFSET 0`` is the documented shape for a locking
    function over a derived row set (same page, "Advisory Locks")."""
    if not imohashes:
        return
    await session.execute(
        text(
            'SELECT pg_advisory_xact_lock(hashtextextended(q.k, 0)) FROM '
            '(SELECT k FROM unnest(CAST(:keys AS text[])) WITH ORDINALITY AS t(k, n) '
            'ORDER BY n OFFSET 0) AS q'
        ),
        {'keys': [_lock_key(project_id, h) for h in imohashes]},
    )


async def rows_for_imohash(
    session: AsyncSession, project_id: uuid.UUID | None, imohash: str
) -> list[ContentRow]:
    """The project's content rows with this ``imohash``, oldest first."""
    rows = await session.execute(
        select(
            ImageContent.id,
            ImageContent.image_id,
            ImageContent.image_path,
            ImageContent.size_bytes,
            ImageContent.content_sha256,
        )
        .where(_owner(project_id), ImageContent.imohash == imohash)
        .order_by(ImageContent.id)
    )
    return [
        ContentRow(r.id, r.image_id, r.image_path, r.size_bytes, r.content_sha256) for r in rows
    ]


async def rows_for_imohashes(
    session: AsyncSession, project_id: uuid.UUID | None, imohashes: Iterable[str]
) -> dict[str, list[ContentRow]]:
    """:func:`rows_for_imohash` for several ``imohash`` values in one query; every wanted value is
    a key (an empty list when no row has it)."""
    wanted = sorted(set(imohashes))
    found: dict[str, list[ContentRow]] = {h: [] for h in wanted}
    if not wanted:
        return found
    rows = await session.execute(
        select(
            ImageContent.imohash,
            ImageContent.id,
            ImageContent.image_id,
            ImageContent.image_path,
            ImageContent.size_bytes,
            ImageContent.content_sha256,
        )
        .where(_owner(project_id), ImageContent.imohash.in_(wanted))
        .order_by(ImageContent.id)
    )
    for r in rows:
        found[r.imohash].append(
            ContentRow(r.id, r.image_id, r.image_path, r.size_bytes, r.content_sha256)
        )
    return found


async def imohashes_with_rows(
    session: AsyncSession, project_id: uuid.UUID | None, imohashes: Iterable[str]
) -> set[str]:
    """Which of ``imohashes`` the project already stores (one query)."""
    wanted = sorted(set(imohashes))
    if not wanted:
        return set()
    rows = await session.execute(
        select(ImageContent.imohash)
        .where(_owner(project_id), ImageContent.imohash.in_(wanted))
        .distinct()
    )
    return {r.imohash for r in rows}


async def set_sha256(session: AsyncSession, row_id: int, sha256: str) -> None:
    """Record a row's SHA-256 once it is known; a row that has one keeps it."""
    await session.execute(
        update(ImageContent)
        .where(ImageContent.id == row_id, ImageContent.content_sha256.is_(None))
        .values(content_sha256=sha256)
    )


async def insert_content(
    session: AsyncSession,
    project_id: uuid.UUID | None,
    *,
    imohash: str,
    sha256: str | None,
    image_id: str,
    image_path: str,
    size_bytes: int,
) -> str | None:
    """Insert the row for ``image_id``; ``None`` when stored now.

    An image that already has a row (a retry after a lost response) answers its own id when the
    row is for the same ``imohash``, and raises :class:`ContentKeyConflictError` for another.
    """
    stored = await session.execute(
        pg_insert(ImageContent)
        .values(
            project_id=project_id,
            imohash=imohash,
            content_sha256=sha256,
            image_id=image_id,
            image_path=image_path,
            size_bytes=size_bytes,
        )
        .on_conflict_do_nothing()
        .returning(ImageContent.id)
    )
    if stored.first() is not None:
        return None
    held = (
        await session.execute(
            select(ImageContent.imohash).where(
                _owner(project_id), ImageContent.image_id == image_id
            )
        )
    ).scalar_one_or_none()
    if held != imohash:
        msg = f'image {image_id} already holds other content'
        raise ContentKeyConflictError(msg)
    return image_id


@dataclass(frozen=True)
class NewContent:
    """One content row to insert."""

    imohash: str
    sha256: str | None
    image_id: str
    image_path: str
    size_bytes: int


async def insert_contents(
    session: AsyncSession, project_id: uuid.UUID | None, rows: Sequence[NewContent]
) -> set[str]:
    """Insert the rows in one statement; the image ids that were inserted. A row that conflicts
    (the image already has one, or its SHA-256 is taken) is skipped: answer it with
    :func:`insert_content`."""
    inserted: set[str] = set()
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start : start + CHUNK]
        stored = await session.execute(
            pg_insert(ImageContent)
            .values(
                [
                    {
                        'project_id': project_id,
                        'imohash': r.imohash,
                        'content_sha256': r.sha256,
                        'image_id': r.image_id,
                        'image_path': r.image_path,
                        'size_bytes': r.size_bytes,
                    }
                    for r in chunk
                ]
            )
            .on_conflict_do_nothing()
            .returning(ImageContent.image_id)
        )
        inserted.update(r.image_id for r in stored)
    return inserted


async def recorded_images(
    session: AsyncSession, project_id: uuid.UUID | None, image_ids: Iterable[str]
) -> set[str]:
    """The image ids that already have a content row."""
    wanted = sorted(set(image_ids))
    if not wanted:
        return set()
    rows = await session.execute(
        select(ImageContent.image_id).where(_owner(project_id), ImageContent.image_id.in_(wanted))
    )
    return {r.image_id for r in rows}


async def release_contents(
    session: AsyncSession, project_id: uuid.UUID | None, image_ids: Iterable[str]
) -> int:
    """Delete the exact-content rows of ``image_ids`` (the image delete path)."""
    ids = sorted(set(image_ids))
    done = 0
    for start in range(0, len(ids), CHUNK):
        result = await session.execute(
            delete(ImageContent).where(
                _owner(project_id), ImageContent.image_id.in_(ids[start : start + CHUNK])
            )
        )
        done += result.rowcount  # type: ignore[attr-defined]
    return done
