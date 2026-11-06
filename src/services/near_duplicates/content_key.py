"""The exact duplicate key: the SHA-256 of the stored bytes, one ``image_contents`` row per image.

The insert is the duplicate check (it replaces the sampled-hash lookup against the index):
two workers that ingest one file at the same time meet at the unique constraint, so the second
one is told which image holds the content instead of racing past an index that has not refreshed
(#358). A row is written before the images document it describes and is removed when the image
is deleted.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from src.db.repos import image_contents as repo
from src.db.session import async_session_factory, project_tx
from src.services.config_store.bound_project import bound_project_id
from src.utils.stage_timing import stage_timer


if TYPE_CHECKING:
    from collections.abc import Iterable


def content_sha256(data: bytes) -> str:
    """Lowercase hex SHA-256 of the whole byte string."""
    with stage_timer('hash', nbytes=len(data)):
        return hashlib.sha256(data).hexdigest()


async def find_bound(sha256s: Iterable[str]) -> dict[str, str]:
    """``content_sha256 -> image_id`` of the bound project's stored contents."""
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        return await repo.find_contents(session, project_id, sha256s)


async def claim_bound(sha256: str, image_id: str, size_bytes: int) -> str | None:
    """Store the content for ``image_id``; ``None`` when stored now, else the holder's image id."""
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        return await repo.claim_content(session, project_id, sha256, image_id, size_bytes)


async def release_bound(image_ids: Iterable[str]) -> int:
    """Forget the contents of deleted (or never written) images of the bound project."""
    ids = list(image_ids)
    if not ids:
        return 0
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        return await repo.release_contents(session, project_id, ids)
