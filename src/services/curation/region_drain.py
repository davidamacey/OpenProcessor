"""BA-3: the region-drain stability verdict for ``GET /ingest/region_drain``.

An ingest walker used to invent its own "stability window" client-side
(wait N polls of ``total_unfinished == 0`` before triggering
``/pipeline/auto_label``) because the backend only ever served the raw
counts. This computes the same verdict server-side so every client
agrees, and so the parameters (``poll_interval_s`` / ``stable_polls``)
are served, not hardcoded twice.

Single-process, in-memory tracker: a zero-streak counter plus the
timestamp the streak started. Good enough for the one walker process
this endpoint is built for; a multi-worker deployment polling this
route from several processes would each track their own streak (no
shared state), which only affects when a client observes ``drained``
flipping true, never whether ``pending_detection``/``pending_verification``
themselves are correct (those are always read fresh from OpenSearch).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime


def region_drain_poll_interval_s() -> float:
    """Read per call (not frozen at import time) so tests and a running
    process both see an env change take effect."""
    return float(os.environ.get('OP_REGION_DRAIN_POLL_INTERVAL_S', '10'))


def region_drain_stable_polls() -> int:
    return int(os.environ.get('OP_REGION_DRAIN_STABLE_POLLS', '3'))


@dataclass
class _StreakState:
    streak: int = 0
    zero_since: datetime | None = None


_state = _StreakState()


@dataclass(frozen=True)
class DrainVerdict:
    drained: bool
    stable_for_s: float
    observed_at: str


def observe_drain(total_unfinished: int, *, now: datetime | None = None) -> DrainVerdict:
    """Update the module-level streak tracker and return the verdict for
    this poll. Call once per ``GET /ingest/region_drain`` request."""
    ts = now or datetime.now(UTC)
    if total_unfinished == 0:
        _state.streak += 1
        if _state.zero_since is None:
            _state.zero_since = ts
    else:
        _state.streak = 0
        _state.zero_since = None
    stable_for_s = (ts - _state.zero_since).total_seconds() if _state.zero_since else 0.0
    drained = total_unfinished == 0 and _state.streak >= region_drain_stable_polls()
    return DrainVerdict(drained=drained, stable_for_s=stable_for_s, observed_at=ts.isoformat())


def _reset_for_tests() -> None:
    _state.streak = 0
    _state.zero_since = None


__all__ = [
    'DrainVerdict',
    'observe_drain',
    'region_drain_poll_interval_s',
    'region_drain_stable_polls',
]
