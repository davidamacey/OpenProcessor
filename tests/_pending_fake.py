"""An in-memory stand-in for the near-duplicate pending writes, for the offline tier."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PendingCalls:
    """Every pending write a test made: ``(level, sorted ids, reason)``. Core ingest's
    ``record_pending`` is recorded at level ``core``."""

    calls: list[tuple[str, list[str], str]] = field(default_factory=list)

    async def enqueue_pending_bound(self, level: str, member_ids: object, reason: str) -> int:
        ids = sorted(member_ids)  # type: ignore[call-overload]
        if ids:
            self.calls.append((level, ids, reason))
        return len(ids)

    async def record_pending(self, image_ids: object, reason: str) -> int:
        ids = sorted(image_ids)  # type: ignore[call-overload]
        if ids:
            self.calls.append(('core', ids, reason))
        return len(ids)
