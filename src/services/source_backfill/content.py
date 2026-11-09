"""The content step: each stored image's ``imohash`` into ``image_contents``.

Images the 0.6.0 upgrade carried over have no content row, so re-ingesting one of them would
not be refused as an exact duplicate (RR-10). The step reads only the sampled regions of each file
(``imohash.hashfile``), never the whole of it, and writes no SHA-256: that is computed later, from
the stored file, the first time another file lands on the same ``imohash``. The step is
idempotent: an image that already has a row is not read again, and two images with identical bytes
each keep a row (the lookup treats them as one content).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import imohash

from src.db.repos import image_contents as repo


if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ContentCounts:
    recorded: int = 0
    already: int = 0
    unreadable: int = 0
    changed: int = 0

    def add(self, other: ContentCounts) -> None:
        self.recorded += other.recorded
        self.already += other.already
        self.unreadable += other.unreadable
        self.changed += other.changed

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def _sample_hash(path: str) -> tuple[str, int] | None:
    """The ``imohash`` and size of the file, or None when the file cannot be read."""
    if not path:
        return None
    try:
        size = Path(path).stat().st_size
        digest: str = imohash.hashfile(path, hexdigest=True)
    except OSError:
        return None
    return digest, size


async def record_contents(
    session: AsyncSession, project_id: uuid.UUID, images: Sequence[tuple[str, str]]
) -> ContentCounts:
    """Record the content of ``images`` (``(image_id, image_path)`` pairs) in one transaction."""
    counts = ContentCounts()
    done = await repo.recorded_images(session, project_id, [i for i, _ in images])
    counts.already = len(done)
    for image_id, path in images:
        if image_id in done:
            continue
        hashed = await asyncio.to_thread(_sample_hash, path)
        if hashed is None:
            counts.unreadable += 1
            continue
        digest, size = hashed
        try:
            await repo.insert_content(
                session,
                project_id,
                imohash=digest,
                sha256=None,
                image_id=image_id,
                image_path=path,
                size_bytes=size,
            )
        except repo.ContentKeyConflictError:
            counts.changed += 1
            continue
        counts.recorded += 1
    return counts
