"""The exact-content key: one ``image_contents`` row per stored image.

The row carries the sampled ``imohash`` of the image's bytes (always) and its SHA-256 (only once a
second file with that ``imohash`` made the full comparison necessary). ``imohash`` is not unique:
two different files can share it, and the SHA-256 tells them apart (UNIQUE per project where it is
set).

Every function takes a session already inside ``project_tx`` and never commits. A lookup and the
insert that follows it must run in one transaction under :func:`lock_imohash`, so concurrent
callers with the same content are decided one after the other, across every process (#358).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.near_dup import ImageContent


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

CHUNK = 4000


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


async def lock_imohash(session: AsyncSession, project_id: uuid.UUID, imohash: str) -> None:
    """``pg_advisory_xact_lock`` on a 64-bit key of (project, imohash), held to the end of the
    transaction. Two imohashes that share a hashed key only wait for each other."""
    await session.execute(
        text('SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))'),
        {'key': f'image_contents:{project_id}:{imohash}'},
    )


async def rows_for_imohash(
    session: AsyncSession, project_id: uuid.UUID, imohash: str
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
        .where(ImageContent.project_id == project_id, ImageContent.imohash == imohash)
        .order_by(ImageContent.id)
    )
    return [
        ContentRow(r.id, r.image_id, r.image_path, r.size_bytes, r.content_sha256) for r in rows
    ]


async def imohashes_with_rows(
    session: AsyncSession, project_id: uuid.UUID, imohashes: Iterable[str]
) -> set[str]:
    """Which of ``imohashes`` the project already stores (one query)."""
    wanted = sorted(set(imohashes))
    if not wanted:
        return set()
    rows = await session.execute(
        select(ImageContent.imohash)
        .where(ImageContent.project_id == project_id, ImageContent.imohash.in_(wanted))
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
    project_id: uuid.UUID,
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
                ImageContent.project_id == project_id, ImageContent.image_id == image_id
            )
        )
    ).scalar_one_or_none()
    if held != imohash:
        msg = f'image {image_id} already holds other content'
        raise ContentKeyConflictError(msg)
    return image_id


async def recorded_images(
    session: AsyncSession, project_id: uuid.UUID, image_ids: Iterable[str]
) -> set[str]:
    """The image ids that already have a content row."""
    wanted = sorted(set(image_ids))
    if not wanted:
        return set()
    rows = await session.execute(
        select(ImageContent.image_id).where(
            ImageContent.project_id == project_id, ImageContent.image_id.in_(wanted)
        )
    )
    return {r.image_id for r in rows}


async def release_contents(
    session: AsyncSession, project_id: uuid.UUID, image_ids: Iterable[str]
) -> int:
    """Delete the exact-content rows of ``image_ids`` (the image delete path)."""
    ids = sorted(set(image_ids))
    done = 0
    for start in range(0, len(ids), CHUNK):
        result = await session.execute(
            delete(ImageContent).where(
                ImageContent.project_id == project_id,
                ImageContent.image_id.in_(ids[start : start + CHUNK]),
            )
        )
        done += result.rowcount  # type: ignore[attr-defined]
    return done
