"""An in-memory stand-in for the near-duplicate pending writes, for the offline tier."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PendingCalls:
    """Every ``enqueue_pending_bound`` call a test made: ``(level, sorted ids, reason)``."""

    calls: list[tuple[str, list[str], str]] = field(default_factory=list)

    async def enqueue_pending_bound(self, level: str, member_ids: object, reason: str) -> int:
        ids = sorted(member_ids)  # type: ignore[call-overload]
        if ids:
            self.calls.append((level, ids, reason))
        return len(ids)
