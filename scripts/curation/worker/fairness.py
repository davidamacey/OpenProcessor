"""Multi-project fairness scheduling for the detection worker
(``scripts/curation/worker/runner.py`` and its stage modules).

See ``docs/design/openprocessor_internal/projects_plan.md`` §5.1.

This module is deliberately self-contained and asyncio-free at the
top level so the deficit round-robin math is directly unit-testable
without spinning up the full producer/consumer pipeline (see
``tests/projects/test_worker_fairness.py``).

Simplifications made against the plan (see PR description / final
report for the full rationale):

* Idle-project probing is a **simple per-project poll with backoff**,
  not a batched ``_msearch``. Each idle project independently tracks
  its own next-due time, so the number of actual OpenSearch calls per
  cycle is bounded by "idle projects due this cycle", not "all idle
  projects" -- this keeps the O(idle) cost claim in the plan (no
  per-cycle blowup) without wiring a new ``_msearch``-shaped call
  into the existing ``_fetch_pending`` helper, which only knows how
  to search one (bound) project's index at a time.
* Per-project pause is a file flag
  (``<project_state_dir>/pipeline_paused.flag``), not a
  ``pipeline.paused`` settings-doc field -- no such field exists yet
  on this branch (re-checked; see task instructions).
* The liveness doc is a small JSON file under
  ``<project_state_dir>/runtime_detection_worker_<host>.json``, not an
  OpenSearch ``configs`` index doc -- no ``IndexRole.CONFIGS`` /
  ``RegionRuntime`` concept exists yet on this branch (re-checked).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.curation._project_worker_utils import is_project_paused, is_region_stage_paused


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from scripts.curation.worker.state import _ItemTask
    from src.config.projects import ProjectRecord

_IDLE_BACKOFF_MIN_S = 5.0
_IDLE_BACKOFF_MAX_S = 60.0


@dataclass
class _ProjectPollState:
    """Per-project poll bookkeeping: idle backoff and current in-flight."""

    interval_s: float = _IDLE_BACKOFF_MIN_S
    next_due_at: float = 0.0
    in_flight: int = 0

    def due(self, now: float) -> bool:
        return now >= self.next_due_at

    def record_empty(self, now: float) -> None:
        """An empty poll: wait ``interval_s``, then double it (capped)."""
        self.next_due_at = now + self.interval_s
        self.interval_s = min(self.interval_s * 2.0, _IDLE_BACKOFF_MAX_S)

    def record_nonempty(self) -> None:
        """A project with work stays due every cycle; its backoff resets."""
        self.interval_s = _IDLE_BACKOFF_MIN_S
        self.next_due_at = 0.0


@dataclass
class CyclePlan:
    """What the producer fetches this cycle for one due project."""

    record: ProjectRecord
    quota: int
    headroom: int


@dataclass
class FairnessScheduler:
    """Deficit round-robin across active projects with per-project
    in-flight caps and idle-poll backoff (projects_plan.md §5.1).

    One instance lives for the worker process; :meth:`plan_cycle` runs
    once per producer cycle.
    """

    _rotation_start: int = 0
    _poll_state: dict[str, _ProjectPollState] = field(default_factory=dict)

    def _state_for(self, slug: str) -> _ProjectPollState:
        return self._poll_state.setdefault(slug, _ProjectPollState())

    def record_fetch_result(self, slug: str, n_fetched: int, *, now: float | None = None) -> None:
        """Feed back one poll's result: empty polls back off, any work
        keeps the project due every cycle."""
        st = self._state_for(slug)
        if n_fetched > 0:
            st.record_nonempty()
        else:
            st.record_empty(time.monotonic() if now is None else now)

    def set_in_flight(self, counts: dict[str, int]) -> None:
        """Replace every project's in-flight count (the producer derives
        them from its in-flight set each cycle)."""
        for slug, st in self._poll_state.items():
            st.in_flight = counts.get(slug, 0)
        for slug, count in counts.items():
            self._state_for(slug).in_flight = count

    def in_flight(self, slug: str) -> int:
        return self._state_for(slug).in_flight

    def poll_interval(self, slug: str) -> float:
        return self._state_for(slug).interval_s

    def plan_cycle(
        self,
        projects: list[ProjectRecord],
        *,
        fetch_n: int,
        queue_max: int,
        now: float | None = None,
    ) -> list[CyclePlan]:
        """This cycle's first-pass fetch plan, in rotation order.

        The rotation start advances one project per cycle. Each due
        project (not backing off, below its in-flight cap
        ``ceil(queue_max / n_active)``) gets ``quota = max(1, fetch_n //
        n_due)``, clipped to its cap headroom. Leftover capacity is
        handed out by :func:`fetch_pending_multi_project`'s second pass,
        which needs this cycle's real fetch results.
        """
        now = time.monotonic() if now is None else now
        if not projects:
            return []
        in_flight_cap = max(1, math.ceil(queue_max / len(projects)))
        start = self._rotation_start % len(projects)
        ordered = projects[start:] + projects[:start]
        self._rotation_start = (start + 1) % len(projects)

        due: list[tuple[ProjectRecord, int]] = []
        for p in ordered:
            st = self._state_for(p.slug)
            headroom = in_flight_cap - st.in_flight
            if st.due(now) and headroom > 0:
                due.append((p, headroom))
        if not due:
            return []
        quota = max(1, fetch_n // len(due))
        return [CyclePlan(record=p, quota=min(quota, h), headroom=h) for p, h in due]


def discover_pollable_projects(all_active: list[ProjectRecord]) -> list[ProjectRecord]:
    """``registry.active_projects()`` without the projects whose pipeline or
    region stage is paused."""
    return [p for p in all_active if not (is_project_paused(p) or is_region_stage_paused(p))]


def _liveness_path(record: ProjectRecord, host: str) -> Path:
    return Path(record.resources.project_state_dir) / f'runtime_detection_worker_{host}.json'


def write_liveness(
    record: ProjectRecord, *, inflight: int, applied: bool, paused: bool, host: str | None = None
) -> None:
    """Write this project's ``runtime:detection_worker:<host>`` liveness
    doc. See module docstring: a per-project JSON file under
    ``project_state_dir``, not an OpenSearch ``configs`` index doc (that
    mechanism does not exist yet on this branch)."""
    host = host or socket.gethostname()
    path = _liveness_path(record, host)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'inflight': inflight,
        'applied': applied,
        'paused': paused,
        'updated_at': time.time(),
        'host': host,
        'project': record.slug,
    }
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


LIVENESS_INTERVAL_S = 15.0


async def liveness_loop(
    projects: Callable[[], list[ProjectRecord]],
    inflight_counts: Callable[[], Mapping[str, int]],
    stop_event: asyncio.Event,
    *,
    interval_s: float = LIVENESS_INTERVAL_S,
) -> None:
    """Refresh every active project's liveness doc on a timer. The
    producer only writes after a fetch that found new items, so a worker
    busy on a long batch (queue full, everything pending already in
    flight, paused) would otherwise go stale and read as idle to
    ``busy.py``."""
    while not stop_event.is_set():
        counts = inflight_counts()
        for record in projects():
            with contextlib.suppress(OSError):
                write_liveness(
                    record,
                    inflight=counts.get(record.slug, 0),
                    applied=True,
                    paused=is_project_paused(record),
                )
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)


def read_liveness(record: ProjectRecord, *, host: str | None = None) -> dict | None:
    host = host or socket.gethostname()
    path = _liveness_path(record, host)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


async def fetch_pending_multi_project(
    opensearch: Any,
    *,
    registry: Any,
    scheduler: FairnessScheduler,
    fetch_n: int,
    queue_max: int,
    exclude_ids: list[str] | None = None,
) -> list[_ItemTask]:
    """One producer cycle across every active, unpaused project
    (projects_plan.md §5.1).

    First pass: each due project fetches its :meth:`FairnessScheduler.
    plan_cycle` quota under its own binding. Second pass (work
    conserving): capacity the first pass left unused goes, in rotation
    order, to projects whose first fetch came back full, up to each
    one's in-flight headroom. Every fetch excludes the in-flight ids and
    everything already fetched this cycle, so an item is queued once.
    """
    from scripts.curation.worker.cascade import _fetch_pending
    from src.config.project_context import bind_project

    active = discover_pollable_projects(registry.active_projects())
    plans = scheduler.plan_cycle(active, fetch_n=fetch_n, queue_max=queue_max)
    excluded = list(exclude_ids or [])
    tasks: list[_ItemTask] = []
    taken: dict[str, int] = {}
    hungry: list[CyclePlan] = []

    async def _fetch(plan: CyclePlan, n: int) -> int:
        with bind_project(plan.record):
            got = await _fetch_pending(
                opensearch, batch_size=n, exclude_ids=excluded, project=plan.record
            )
        tasks.extend(got)
        excluded.extend(t.crop_id for t in got)
        taken[plan.record.slug] = taken.get(plan.record.slug, 0) + len(got)
        return len(got)

    for plan in plans:
        n = await _fetch(plan, plan.quota)
        scheduler.record_fetch_result(plan.record.slug, n)
        if n >= plan.quota:
            hungry.append(plan)

    while hungry and len(tasks) < fetch_n:
        plan = hungry.pop(0)
        want = min(fetch_n - len(tasks), plan.headroom - taken[plan.record.slug])
        if want <= 0:
            continue
        if await _fetch(plan, want) >= want:
            hungry.append(plan)
    return tasks
