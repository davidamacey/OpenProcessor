"""The exact-content key: one ``image_contents`` row per stored image (SHA-256 of its bytes).

Every function takes a session already inside ``project_tx`` and never commits. The unique
constraints are the duplicate check: a concurrent insert of the same content waits on the index,
so exactly one caller stores it (#358).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.near_dup import ImageContent


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

CHUNK = 4000


class ContentKeyConflictError(ValueError):
    """The image already has a content row for other bytes (its file changed under it)."""


async def claim_content(
    session: AsyncSession, project_id: uuid.UUID, sha256: str, image_id: str, size_bytes: int
) -> str | None:
    """Insert the exact-content row; ``None`` when this call stored it, else the image id that
    already holds the content (``ContentKeyConflictError`` when ``image_id`` holds other content)."""
    stored = await session.execute(
        pg_insert(ImageContent)
        .values(
            project_id=project_id, content_sha256=sha256, image_id=image_id, size_bytes=size_bytes
        )
        .on_conflict_do_nothing()
        .returning(ImageContent.id)
    )
    if stored.first() is not None:
        return None
    holder = (
        await session.execute(
            select(ImageContent.image_id).where(
                ImageContent.project_id == project_id, ImageContent.content_sha256 == sha256
            )
        )
    ).scalar_one_or_none()
    if holder is None:
        msg = f'image {image_id} already holds other content'
        raise ContentKeyConflictError(msg)
    return holder


async def find_contents(
    session: AsyncSession, project_id: uuid.UUID, sha256s: Iterable[str]
) -> dict[str, str]:
    """``content_sha256 -> image_id`` for the contents the project already stores."""
    wanted = sorted(set(sha256s))
    if not wanted:
        return {}
    rows = await session.execute(
        select(ImageContent.content_sha256, ImageContent.image_id).where(
            ImageContent.project_id == project_id, ImageContent.content_sha256.in_(wanted)
        )
    )
    return {r.content_sha256: r.image_id for r in rows}


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
