"""Photos: insert, load, the photo lock and the imohash advisory locks (15a sections 2.1 and 7).

Every function takes a session already inside ``project_tx`` and never commits. The wire id of a
photo is its content-derived ``image_id`` (32 hex digits); the repository converts to and from the
``uuid`` column through :mod:`src.domain.wire_ids`.

The duplicate check of ingest is the unique key itself: ``insert_photos`` answers the photos it
stored, and a photo that already existed is absent from the answer (``ON CONFLICT DO NOTHING``).
The sorted advisory locks on the imohash are the ones ``image_contents`` takes today (#358): one copy,
imported from there until the ingest lane drops that table and moves them here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.photos import Photo
from src.db.repos.image_contents import lock_imohash, lock_imohashes
from src.db.repos.upserts import any_of
from src.domain.wire_ids import to_uuid, to_wire


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

CHUNK = 2000

__all__ = [
    'NewPhoto',
    'PhotoRow',
    'insert_photos',
    'load_photos',
    'lock_imohash',
    'lock_imohashes',
    'lock_photos',
    'photo_keys_by_image_id',
    'set_content_sha256',
]


@dataclass(frozen=True)
class NewPhoto:
    image_id: str
    imohash: bytes
    storage_key: str
    size_bytes: int
    width: int
    height: int
    frame: int
    decode_route: str
    source: str
    source_identifier: str | None = None
    ingest_run_tag: str | None = None
    ingest_job_id: uuid.UUID | None = None
    ingest_request_id: str | None = None


@dataclass(frozen=True)
class PhotoRow:
    photo_key: int
    image_id: str
    imohash: bytes
    content_sha256: bytes | None
    storage_key: str
    size_bytes: int
    width: int
    height: int
    frame: int
    source_state: str
    decode_route: str
    source: str


def _values(project_id: uuid.UUID, photo: NewPhoto) -> dict[str, object]:
    return {
        'project_id': project_id,
        'image_id': to_uuid(photo.image_id),
        'imohash': photo.imohash,
        'storage_key': photo.storage_key,
        'size_bytes': photo.size_bytes,
        'width': photo.width,
        'height': photo.height,
        'frame': photo.frame,
        'decode_route': photo.decode_route,
        'source': photo.source,
        'source_identifier': photo.source_identifier,
        'ingest_run_tag': photo.ingest_run_tag,
        'ingest_job_id': photo.ingest_job_id,
        'ingest_request_id': photo.ingest_request_id,
    }


async def insert_photos(
    session: AsyncSession, project_id: uuid.UUID, photos: Sequence[NewPhoto]
) -> dict[str, int]:
    """Insert the photos; the wire id to ``photo_key`` of those stored now. A photo that already
    exists (a retry, or the same bytes again) is not in the answer and is left as it was."""
    stored: dict[str, int] = {}
    for start in range(0, len(photos), CHUNK):
        result = await session.execute(
            pg_insert(Photo)
            .values([_values(project_id, p) for p in photos[start : start + CHUNK]])
            .on_conflict_do_nothing(index_elements=['project_id', 'image_id'])
            .returning(Photo.photo_key, Photo.image_id)
        )
        stored.update({to_wire(r.image_id): r.photo_key for r in result})
    return stored


async def photo_keys_by_image_id(
    session: AsyncSession, project_id: uuid.UUID, image_ids: Iterable[str]
) -> dict[str, int]:
    """The photos of the project among ``image_ids``, by wire id (one query, key order)."""
    wanted = sorted({to_uuid(i) for i in image_ids})
    if not wanted:
        return {}
    rows = await session.execute(
        select(Photo.image_id, Photo.photo_key)
        .where(Photo.project_id == project_id, any_of(Photo.image_id, wanted))
        .order_by(Photo.photo_key)
    )
    return {to_wire(r.image_id): r.photo_key for r in rows}


async def load_photos(
    session: AsyncSession, project_id: uuid.UUID, photo_keys: Iterable[int]
) -> list[PhotoRow]:
    wanted = sorted(set(photo_keys))
    if not wanted:
        return []
    rows = await session.execute(
        select(Photo)
        .where(Photo.project_id == project_id, any_of(Photo.photo_key, wanted))
        .order_by(Photo.photo_key)
    )
    return [
        PhotoRow(
            p.photo_key, to_wire(p.image_id), p.imohash, p.content_sha256, p.storage_key,
            p.size_bytes, p.width, p.height, p.frame, p.source_state, p.decode_route, p.source,
        )
        for p in rows.scalars()
    ]  # fmt: skip


async def lock_photos(session: AsyncSession, photo_keys: Iterable[int]) -> list[int]:
    """``FOR NO KEY UPDATE`` on the photos, in key order (rank 5 of the lock order, 15a section 7).

    It does not conflict with the ``FOR KEY SHARE`` a child row's foreign-key check takes, so a
    writer that holds a photo never blocks an insert of its items' results. The rows locked: a key
    that is missing was deleted meanwhile (the caller skips it or answers 404).
    """
    wanted = sorted(set(photo_keys))
    if not wanted:
        return []
    rows = await session.execute(
        select(Photo.photo_key)
        .where(any_of(Photo.photo_key, wanted))
        .order_by(Photo.photo_key)
        .with_for_update(key_share=True)
    )
    return [r.photo_key for r in rows]


async def set_content_sha256(session: AsyncSession, photo_key: int, sha256: bytes) -> bool:
    """Record the SHA-256 once an imohash match made the full comparison necessary; a photo that
    has one keeps it."""
    result = await session.execute(
        update(Photo)
        .where(Photo.photo_key == photo_key, Photo.content_sha256.is_(None))
        .values(content_sha256=sha256)
    )
    return bool(result.rowcount)  # type: ignore[attr-defined]
