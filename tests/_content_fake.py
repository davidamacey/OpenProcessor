"""An in-memory stand-in for the exact-content table, for the offline tier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.near_duplicates.content_key import (
    ContentKeyChangedError,
    CoreClaims,
    Verdict,
    content_image_id,
    content_sha256,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.near_duplicates.content_key import CoreEntry


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
        self.core_rows: dict[str, tuple[str, str, bytes | None]] = {}
        """The core index's rows: key to ``(image id, path, bytes)``; a row a test seeds has no
        bytes, so a file with its ``imohash`` is a duplicate of it."""

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

    async def record_copy_bound(
        self, imohash: str, image_id: str, _image_path: str, _size: int
    ) -> None:
        self.rows.setdefault(imohash, image_id)

    async def release_bound(self, image_ids: object) -> int:
        gone = set(image_ids)  # type: ignore[call-overload]
        before = len(self.rows)
        self.rows = {k: i for k, i in self.rows.items() if i not in gone}
        self.files = {i: b for i, b in self.files.items() if i not in gone}
        return before - len(self.rows)

    def _core_decide(self, imohash: str, data: bytes) -> Verdict:
        held = self.core_rows.get(imohash)
        if held is None:
            return Verdict(None, imohash, None)
        sha = content_sha256(data)
        if held[2] is None or content_sha256(held[2]) == sha:
            return Verdict(held[0], imohash, sha, held[1])
        other = self.core_rows.get(sha)
        return Verdict(other[0] if other else None, sha, sha, other[1] if other else None)

    async def resolve_many_core(self, entries: Sequence[tuple[str, bytes]]) -> list[Verdict]:
        return [self._core_decide(h, data) for h, data in entries]

    async def claim_many_core(self, entries: Sequence[CoreEntry]) -> CoreClaims:
        duplicates: dict[str, Verdict] = {}
        conflicts: dict[str, str] = {}
        for entry in sorted(entries, key=lambda e: e.imohash):
            verdict = self._core_decide(entry.imohash, entry.data)
            if verdict.holder is not None:
                duplicates[entry.image_id] = verdict
                continue
            if any(row[0] == entry.image_id for row in self.core_rows.values()):
                conflicts[entry.image_id] = f'image {entry.image_id} already holds other content'
                continue
            self.core_rows[verdict.key] = (entry.image_id, entry.image_path, entry.data)
        return CoreClaims(duplicates, conflicts)

    async def release_core(self, image_ids: object) -> int:
        gone = set(image_ids)  # type: ignore[call-overload]
        before = len(self.core_rows)
        self.core_rows = {k: v for k, v in self.core_rows.items() if v[0] not in gone}
        return before - len(self.core_rows)
