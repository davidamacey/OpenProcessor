"""BA-3: the region-drain stability verdict for ``GET /ingest/region_drain``.

An ingest walker used to invent its own "stability window" client-side
(wait N polls of ``total_unfinished == 0`` before triggering
``/pipeline/auto_label``) because the backend only ever served the raw
counts. This computes the same verdict server-side so every client
agrees, and so the parameters (``poll_interval_s`` / ``stable_polls``)
are served, not hardcoded twice.

**Multi-worker correctness (2026-09-25 fix).** ``yolo-api`` runs under
``--workers=N`` -- separate OS processes. This used to keep the streak
counter (``_StreakState``) in module memory: an ingest walker polling
this endpoint every ``poll_interval_s`` lands on essentially a random
worker process each time, so the streak reset on almost every poll and
the ``drained`` verdict flapped depending purely on which process
happened to answer -- never truly reaching ``stable_polls`` consecutive
zeros from any single process's point of view. Fixed by persisting the
same ``(streak, zero_since)`` pair to a small state.json under
``OP_REGION_DRAIN_STATE_DIR`` (shared ``/jobs`` volume every worker
mounts, same convention as the file-backed job runners in this package),
atomically written on every call. This is a per-poll read-modify-write,
not append-only, so two polls landing on different processes at the
exact same instant could race and one increment could be lost -- that is
an acceptable, self-correcting inaccuracy (the next poll fixes the count
within one more ``poll_interval_s``), not a correctness issue: it only
ever delays ``drained`` flipping true by at most one extra poll, and
``pending_detection``/``pending_verification`` themselves are always read
fresh from OpenSearch regardless.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def region_drain_poll_interval_s() -> float:
    """Read per call (not frozen at import time) so tests and a running
    process both see an env change take effect."""
    return float(os.environ.get('OP_REGION_DRAIN_POLL_INTERVAL_S', '10'))


def region_drain_stable_polls() -> int:
    return int(os.environ.get('OP_REGION_DRAIN_STABLE_POLLS', '3'))


def _state_dir() -> Path:
    """Resolved fresh each call so tests can override via monkeypatch
    (same convention as the file-backed job runners' ``_state_dir`` /
    ``_jobs_dir`` helpers)."""
    return Path(os.environ.get('OP_REGION_DRAIN_STATE_DIR', '/jobs/region_drain'))


def _state_file() -> Path:
    return _state_dir() / 'state.json'


@dataclass
class _StreakState:
    streak: int = 0
    zero_since: float | None = None  # epoch seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_state() -> _StreakState:
    try:
        raw = json.loads(_state_file().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return _StreakState()
    state = _StreakState()
    for k, v in raw.items():
        if hasattr(state, k):
            setattr(state, k, v)
    return state


def _atomic_write(state: _StreakState) -> None:
    _state_dir().mkdir(parents=True, exist_ok=True)
    tmp = _state_file().with_suffix('.tmp')
    tmp.write_text(json.dumps(state.to_dict()))
    tmp.replace(_state_file())


@dataclass(frozen=True)
class DrainVerdict:
    drained: bool
    stable_for_s: float
    observed_at: str


def observe_drain(total_unfinished: int, *, now: datetime | None = None) -> DrainVerdict:
    """Update the persisted streak tracker and return the verdict for
    this poll. Call once per ``GET /ingest/region_drain`` request, from
    any worker process -- the state lives on shared disk, not in this
    process's memory."""
    ts = now or datetime.now(UTC)
    epoch = ts.timestamp()
    state = _read_state()
    if total_unfinished == 0:
        state.streak += 1
        if state.zero_since is None:
            state.zero_since = epoch
    else:
        state.streak = 0
        state.zero_since = None
    _atomic_write(state)

    stable_for_s = max(0.0, epoch - state.zero_since) if state.zero_since is not None else 0.0
    drained = total_unfinished == 0 and state.streak >= region_drain_stable_polls()
    return DrainVerdict(drained=drained, stable_for_s=stable_for_s, observed_at=ts.isoformat())


def _reset_for_tests() -> None:
    with contextlib.suppress(FileNotFoundError):
        _state_file().unlink()


__all__ = [
    'DrainVerdict',
    'observe_drain',
    'region_drain_poll_interval_s',
    'region_drain_stable_polls',
]
