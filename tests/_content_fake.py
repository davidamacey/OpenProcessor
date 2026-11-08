"""An in-memory stand-in for the exact-content table, for the offline tier."""

from __future__ import annotations


class ContentRows:
    """``content_sha256 -> image_id`` of one test, with the three bound calls the code makes."""

    def __init__(self) -> None:
        self.rows: dict[str, str] = {}

    async def find_bound(self, sha256s: object) -> dict[str, str]:
        return {h: self.rows[h] for h in sha256s if h in self.rows}  # type: ignore[attr-defined]

    async def claim_bound(self, sha256: str, image_id: str, _size_bytes: int) -> str | None:
        holder = self.rows.get(sha256)
        if holder is not None:
            return holder
        self.rows[sha256] = image_id
        return None

    async def release_bound(self, image_ids: object) -> int:
        gone = set(image_ids)  # type: ignore[call-overload]
        before = len(self.rows)
        self.rows = {h: i for h, i in self.rows.items() if i not in gone}
        return before - len(self.rows)
