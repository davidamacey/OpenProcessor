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


async def _default_check_segmenter_health() -> tuple[bool, str]:
    """``GET {OP_SEGMENTER_URL}/health`` -- delegates to the same probe
    ``GET /curation/models/status`` already uses for the segmenter roster
    entry (:func:`src.routers.curation._models_segmenter._segmenter_health`),
    rather than re-implementing an HTTP health check here. That probe
    hits the segmenter's own HTTP service directly and requires
    ``loaded: true`` -- unlike the Triton repository index, which SAM 3 (or
    whatever's configured) never appears in, since it isn't a Triton-served
    model (V-1 follow-up)."""
    from src.routers.curation._models_segmenter import _segmenter_health

    url = os.environ.get('OP_SEGMENTER_URL', '').strip()
    if not url:
        return False, 'OP_SEGMENTER_URL is not configured'
    # OP_SEGMENTER_URLS may load-balance across several hosts; one healthy
    # host is enough to characterize the dependency as reachable, same as
    # the /models/status roster probe.
    first_url = url.split(',')[0].strip().rstrip('/')
    status, last_error = await _segmenter_health(first_url)
    if status == 'ready':
        return True, f'{first_url}/health loaded=true'
    return False, last_error or f'{first_url}/health unavailable'


async def check_region_dependencies(
    get_repository_index: Callable[[], Awaitable[list[dict[str, Any]]]],
    profile: DetectionProfile,
    *,
    now: datetime | None = None,
    check_segmenter_health: Callable[[], Awaitable[tuple[bool, str]]] | None = None,
) -> list[RegionDependencyStatus]:
    """Check the active profile's detector/segmenter dependencies.

    The detector is Triton-served, so it's checked against Triton's
    repository index (``get_repository_index``). The segmenter (e.g.
    SAM 3) runs as its own HTTP service, not in Triton -- it's checked via
    ``check_segmenter_health`` (defaults to :func:`_default_check_segmenter_health`,
    a ``GET {OP_SEGMENTER_URL}/health`` with a short timeout, requiring
    ``loaded: true``). Looking the segmenter up in the Triton repository
    index (the original implementation) meant ``stall_reason`` never
    cleared even with a perfectly healthy segmenter, since it would never
    appear in that index.

    Returns one :class:`RegionDependencyStatus` per configured (non-empty)
    dependency, in ``(role, model)`` stable order -- detector first, then
    segmenter. Empty when the profile has no ``detector_model`` at all (no
    active profile -- nothing to check; the neutral/off case, not a stall).
    """
    if not profile.detector_model:
        return []

    ts = now or datetime.now(UTC)
    epoch = ts.timestamp()
    tracked = _read_state()
    results: list[RegionDependencyStatus] = []

    # ---- detector: Triton repository index ----
    model = profile.detector_model
    try:
        repo_index = await get_repository_index()
    except Exception as exc:
        # Triton itself unreachable from the API -- unknown-unavailable,
        # not silently "ready".
        since = tracked.setdefault(model, epoch)
        results.append(
            RegionDependencyStatus(
                role='detector',
                model=model,
                ready=False,
                unavailable_since=datetime.fromtimestamp(since, tz=UTC).isoformat(),
                detail=f'Triton repository index unavailable: {exc}',
            )
        )
    else:
        states = _model_state_map(repo_index)
        triton_state = states.get(model)
        ready = triton_state == 'READY'
        if ready:
            tracked.pop(model, None)
            results.append(
                RegionDependencyStatus(
                    role='detector',
                    model=model,
                    ready=True,
                    unavailable_since=None,
                    detail='READY',
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
                    role='detector',
                    model=model,
                    ready=False,
                    unavailable_since=datetime.fromtimestamp(since, tz=UTC).isoformat(),
                    detail=detail,
                )
            )

    # ---- segmenter: its own HTTP service, never Triton ----
    if profile.segmenter_name:
        model = profile.segmenter_name
        checker = check_segmenter_health or _default_check_segmenter_health
        try:
            ready, detail = await checker()
        except Exception as exc:  # a broken checker must not 500 the caller
            ready, detail = False, f'segmenter health check raised: {exc}'
        if ready:
            tracked.pop(model, None)
            results.append(
                RegionDependencyStatus(
                    role='segmenter',
                    model=model,
                    ready=True,
                    unavailable_since=None,
                    detail=detail,
                )
            )
        else:
            since = tracked.setdefault(model, epoch)
            results.append(
                RegionDependencyStatus(
                    role='segmenter',
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
    # The cascade falls through to the segmenter when the detector is missing
    # (segmenter-only deployments), so a down detector only stalls the queue
    # when no ready segmenter can take over. A down segmenter still stalls
    # the items the detector misses.
    if any(d.role == 'segmenter' and d.ready for d in dependencies):
        unavailable = [d for d in unavailable if d.role != 'detector']
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
