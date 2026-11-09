"""An in-memory stand-in for the exact-content table, for the offline tier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.near_duplicates.content_key import (
    ContentKeyChangedError,
    Verdict,
    content_image_id,
    content_sha256,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


class ContentRows:
    """One test's rows with the bound calls the code makes.

    ``rows`` maps a key to an image id: the ``imohash`` of the first image with it, and the
    SHA-256 of a file that collided with a stored ``imohash``. A row seeded by a test has no
    bytes, so a file with its ``imohash`` is a duplicate of it; a row a claim wrote keeps the
    bytes, so the comparison the real key makes by SHA-256 happens here too.
    """

    def __init__(self) -> None:
        self.rows: dict[str, str] = {}
        self.files: dict[str, bytes] = {}

    def _decide(self, imohash: str, data: bytes) -> Verdict:
        holder = self.rows.get(imohash)
        if holder is None:
            return Verdict(None, imohash, None)
        stored = self.files.get(holder)
        sha = content_sha256(data)
        if stored is None or content_sha256(stored) == sha:
            return Verdict(holder, imohash, sha)
        return Verdict(self.rows.get(sha), sha, sha)

    async def resolve_many_bound(self, entries: Sequence[tuple[str, bytes]]) -> list[Verdict]:
        return [self._decide(h, data) for h, data in entries]

    async def claim_bound(
        self, imohash: str, image_id: str, _image_path: str, data: bytes
    ) -> str | None:
        verdict = self._decide(imohash, data)
        if verdict.holder is not None:
            return verdict.holder
        if content_image_id(verdict.key) != image_id:
            raise ContentKeyChangedError(image_id)
        self.rows[verdict.key] = image_id
        self.files[image_id] = data
        return None

    async def release_bound(self, image_ids: object) -> int:
        gone = set(image_ids)  # type: ignore[call-overload]
        before = len(self.rows)
        self.rows = {k: i for k, i in self.rows.items() if i not in gone}
        self.files = {i: b for i, b in self.files.items() if i not in gone}
        return before - len(self.rows)
