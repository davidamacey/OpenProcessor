"""V-1 (fresh-start E2E findings 2026-09-25, coordinator visual review):
surface *why* the region-detection queue is stalled.

With a region profile configured but its Triton model(s) unreachable
(segmenter down, no `HF_TOKEN`, model never loaded, ...), items sit in
`pending_detection` forever -- by design, the detection worker never
writes a terminal status on an infra failure (see
`scripts/curation/worker/cascade.py`'s `_process_crop` docstring: doing
so would lock a crop out of the workflow once the underlying issue is
fixed). That's the right item-level behavior -- items stay retryable
with no code change needed here -- but the operator-facing surface
(`GET /ingest/region_drain`, `/stats/dataset`) had no way to tell
"stalled because nothing is running yet" apart from "stalled because a
dependency is down", both of which just read as a flat pending count
with no trend.

This checks the active profile's Triton model(s) directly from the API
process (which can always reach Triton over the network, unlike probing
the detection worker container, whose own heartbeat is written to a
container-local path the API can't see) and persists a per-model
"first seen unavailable" timestamp to the same shared `/jobs` volume
`region_drain.py`'s streak state uses, so repeated polls report a
stable `unavailable_since`, not "just now" every time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.config import DetectionProfile


def _state_dir() -> Path:
    """Resolved fresh each call (same convention as region_drain.py's
    _state_dir) so tests can override via monkeypatch/env."""
    return Path(os.environ.get('OP_REGION_DRAIN_STATE_DIR', '/jobs/region_drain'))


def _state_file() -> Path:
    return _state_dir() / 'dependency_health.json'


def _read_state() -> dict[str, float]:
    try:
        raw = json.loads(_state_file().read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def _atomic_write(state: dict[str, float]) -> None:
    """Best-effort persist. A write failure (state dir not writable,
    volume not mounted) must never turn "what's the region-drain stall
    reason" into a 500 on /stats/dataset or /ingest/region_drain --
    the caller just re-derives "since" as "now" on the next successful
    poll instead of losing the whole response."""
    try:
        _state_dir().mkdir(parents=True, exist_ok=True)
        tmp = _state_file().with_suffix('.tmp')
        tmp.write_text(json.dumps(state))
        tmp.replace(_state_file())
    except OSError:
        pass


@dataclass(frozen=True)
class RegionDependencyStatus:
    """One Triton model the active region profile depends on."""

    role: str  # 'detector' | 'segmenter'
    model: str  # Triton model name
    ready: bool
    unavailable_since: str | None  # ISO8601, None when ready
    detail: str


def _model_state_map(repo_index: list[dict[str, Any]]) -> dict[str, str]:
    """``{model_name: state}`` from a Triton repository-index response.
    A model absent from the index entirely (never loaded) has no entry."""
    return {
        str(entry.get('name', '')): str(entry.get('state', ''))
        for entry in repo_index
        if entry.get('name')
    }


async def check_region_dependencies(
    get_repository_index: Callable[[], Awaitable[list[dict[str, Any]]]],
    profile: DetectionProfile,
    *,
    now: datetime | None = None,
) -> list[RegionDependencyStatus]:
    """Check the active profile's detector/segmenter Triton models.

    Returns one :class:`RegionDependencyStatus` per configured (non-empty)
    model name, in ``(role, model)`` stable order. Empty when the profile
    has no ``detector_model`` at all (no active profile -- nothing to
    check; the neutral/off case, not a stall).
    """
    if not profile.detector_model:
        return []

    candidates: list[tuple[str, str]] = [('detector', profile.detector_model)]
    if profile.segmenter_name:
        candidates.append(('segmenter', profile.segmenter_name))

    try:
        repo_index = await get_repository_index()
    except Exception as exc:
        # Triton itself unreachable from the API -- every configured
        # dependency is unknown-unavailable, not silently "ready".
        ts = now or datetime.now(UTC)
        epoch = ts.timestamp()
        state = _read_state()
        results = []
        for role, model in candidates:
            since = state.setdefault(model, epoch)
            results.append(
                RegionDependencyStatus(
                    role=role,
                    model=model,
                    ready=False,
                    unavailable_since=datetime.fromtimestamp(since, tz=UTC).isoformat(),
                    detail=f'Triton repository index unavailable: {exc}',
                )
            )
        _atomic_write(state)
        return results

    states = _model_state_map(repo_index)
    ts = now or datetime.now(UTC)
    epoch = ts.timestamp()
    tracked = _read_state()
    results = []
    for role, model in candidates:
        triton_state = states.get(model)
        ready = triton_state == 'READY'
        if ready:
            tracked.pop(model, None)
            results.append(
                RegionDependencyStatus(
                    role=role, model=model, ready=True, unavailable_since=None, detail='READY'
                )
            )
        else:
            since = tracked.setdefault(model, epoch)
            detail = (
                f'Triton reports state={triton_state!r}'
                if triton_state is not None
                else 'not in Triton repository index (never loaded)'
            )
            results.append(
                RegionDependencyStatus(
                    role=role,
                    model=model,
                    ready=False,
                    unavailable_since=datetime.fromtimestamp(since, tz=UTC).isoformat(),
                    detail=detail,
                )
            )
    _atomic_write(tracked)
    return results


def stall_reason(
    dependencies: list[RegionDependencyStatus], *, pending_detection: int
) -> str | None:
    """A single human-readable reason string for the dashboard, or None
    when there's nothing to explain (nothing pending, or every
    dependency is ready)."""
    if pending_detection <= 0:
        return None
    unavailable = [d for d in dependencies if not d.ready]
    if not unavailable:
        return None
    parts = [f'{d.role} ({d.model}) unavailable since {d.unavailable_since}' for d in unavailable]
    return f'{pending_detection} item(s) awaiting region detection; ' + '; '.join(parts)


def _reset_for_tests() -> None:
    import contextlib

    with contextlib.suppress(FileNotFoundError):
        _state_file().unlink()


__all__ = [
    'RegionDependencyStatus',
    'check_region_dependencies',
    'stall_reason',
]
