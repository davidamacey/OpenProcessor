"""The region-drain stability verdict for ``GET /ingest/region_drain``.

An ingest walker used to invent its own "stability window" client-side
(wait N polls of ``total_unfinished == 0`` before triggering
``/pipeline/auto_label``) because the backend only ever served the raw
counts. This computes the same verdict server-side so every client
agrees, and so the parameters (``poll_interval_s`` / ``stable_polls``)
are served, not hardcoded twice.

``api`` runs many worker processes and a walker's polls land on any of them, so the
``(streak, zero_since)`` pair lives in the project's ``region_drain`` marker
(``project_state``), updated with optimistic concurrency on its revision: two polls at the same
instant retry instead of losing an increment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.env import env_float, env_int
from src.config.project_context import current_project_id
from src.db.repos.records import modify_project_state
from src.db.session import project_tx


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_KEY = 'region_drain'


def region_drain_poll_interval_s() -> float:
    """Read per call (not frozen at import time) so tests and a running
    process both see an env change take effect."""
    return env_float('OP_REGION_DRAIN_POLL_INTERVAL_S', 10)


def region_drain_stable_polls() -> int:
    return env_int('OP_REGION_DRAIN_STABLE_POLLS', 3)


@dataclass(frozen=True)
class DrainVerdict:
    drained: bool
    stable_for_s: float
    observed_at: str


async def observe_drain(
    session: AsyncSession, total_unfinished: int, *, now: datetime | None = None
) -> DrainVerdict:
    """Update the persisted streak tracker and return the verdict for this poll.

    Call once per ``GET /ingest/region_drain`` request, from any worker process: the state is
    the project's ``region_drain`` row, not this process's memory.
    """
    ts = now or datetime.now(UTC)
    epoch = ts.timestamp()

    def advance(current: dict[str, Any] | None) -> dict[str, Any]:
        streak = int((current or {}).get('streak', 0))
        zero_since = (current or {}).get('zero_since')
        if total_unfinished == 0:
            return {'streak': streak + 1, 'zero_since': epoch if zero_since is None else zero_since}
        return {'streak': 0, 'zero_since': None}

    project_id = current_project_id()
    async with project_tx(session, project_id):
        state = await modify_project_state(session, project_id, _KEY, advance)

    zero_since = state['zero_since']
    stable_for_s = max(0.0, epoch - zero_since) if zero_since is not None else 0.0
    drained = total_unfinished == 0 and state['streak'] >= region_drain_stable_polls()
    return DrainVerdict(drained=drained, stable_for_s=stable_for_s, observed_at=ts.isoformat())


__all__ = [
    'DrainVerdict',
    'observe_drain',
    'region_drain_poll_interval_s',
    'region_drain_stable_polls',
]
