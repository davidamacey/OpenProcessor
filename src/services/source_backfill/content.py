"""The content step: one read of each stored image's file, its SHA-256 into ``image_contents``.

Images the 0.6.0 upgrade carried over have no content row, so re-ingesting one of them would
not be refused as an exact duplicate (RR-10). The step is idempotent: an image that already has
a row is not read again, and a second image with identical bytes keeps no row of its own (the
first image holds the content, as at ingest).
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.db.repos import image_contents as repo


if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ContentCounts:
    recorded: int = 0
    already: int = 0
    same_bytes_as_another: int = 0
    unreadable: int = 0
    changed: int = 0

    def add(self, other: ContentCounts) -> None:
        self.recorded += other.recorded
        self.already += other.already
        self.same_bytes_as_another += other.same_bytes_as_another
        self.unreadable += other.unreadable
        self.changed += other.changed

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def _read_and_hash(path: str) -> tuple[str, int] | None:
    """The SHA-256 and size of the file's bytes, or None when the file cannot be read."""
    if not path:
        return None
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest(), len(data)


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
        hashed = await asyncio.to_thread(_read_and_hash, path)
        if hashed is None:
            counts.unreadable += 1
            continue
        sha256, size = hashed
        try:
            holder = await repo.claim_content(session, project_id, sha256, image_id, size)
        except repo.ContentKeyConflictError:
            counts.changed += 1
            continue
        if holder is None:
            counts.recorded += 1
        else:
            counts.same_bytes_as_another += 1
    return counts
