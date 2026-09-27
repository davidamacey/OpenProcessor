"""Multi-project fairness scheduling for the detection worker
(``scripts/curation/worker/runner.py``).

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

import json
import math
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord

PIPELINE_PAUSED_FLAG_NAME = 'pipeline_paused.flag'

_IDLE_BACKOFF_MIN_S = 5.0
_IDLE_BACKOFF_MAX_S = 60.0


def is_project_paused(record: ProjectRecord) -> bool:
    """A project is paused iff ``<project_state_dir>/pipeline_paused.flag``
    exists. Minimal primitive -- no HTTP route in this task (see module
    docstring); a later task can add one that just writes/removes this
    file."""
    return (record.resources.project_state_dir / PIPELINE_PAUSED_FLAG_NAME).exists()


@dataclass
class _ProjectPollState:
    """Per-project idle-backoff bookkeeping."""

    interval_s: float = _IDLE_BACKOFF_MIN_S
    next_due_at: float = 0.0
    in_flight: int = 0

    def due(self, now: float) -> bool:
        return now >= self.next_due_at

    def record_empty(self, now: float) -> None:
        """Back off: double the interval, capped."""
        self.interval_s = min(self.interval_s * 2.0, _IDLE_BACKOFF_MAX_S)
        self.next_due_at = now + self.interval_s

    def record_nonempty(self, now: float) -> None:
        """Reset to the fast interval on any non-empty result."""
        self.interval_s = _IDLE_BACKOFF_MIN_S
        self.next_due_at = now + self.interval_s


@dataclass
class CyclePlan:
    """What the producer should do this cycle for one project."""

    slug: str
    record: ProjectRecord
    quota: int
    in_flight_cap: int
    should_poll: bool


@dataclass
class FairnessScheduler:
    """Deficit round-robin, work-conserving scheduler across active
    projects, plus per-project in-flight caps and idle-poll backoff.

    One instance lives for the worker process's lifetime; ``plan_cycle``
    is called once per producer cycle.
    """

    _rotation_start: int = 0
    _poll_state: dict[str, _ProjectPollState] = field(default_factory=dict)

    def _state_for(self, slug: str) -> _ProjectPollState:
        st = self._poll_state.get(slug)
        if st is None:
            st = _ProjectPollState(next_due_at=0.0)
            self._poll_state[slug] = st
        return st

    def record_fetch_result(self, slug: str, n_fetched: int, *, now: float | None = None) -> None:
        """Feed back how many items a project's fetch returned this
        cycle, so idle projects back off and active ones stay fast."""
        now = time.monotonic() if now is None else now
        st = self._state_for(slug)
        if n_fetched > 0:
            st.record_nonempty(now)
        else:
            st.record_empty(now)

    def set_in_flight(self, slug: str, count: int) -> None:
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
        backlog_hint: dict[str, int] | None = None,
        now: float | None = None,
    ) -> list[CyclePlan]:
        """Decide, for this cycle, which projects to poll and with what
        quota / in-flight cap.

        ``backlog_hint`` is unused here -- the real work-conserving
        second pass needs *this* cycle's actual fetch results (a project
        that asked for its quota but had fewer items available), which
        aren't known until after the first-pass fetch runs. That pass
        lives in :func:`scripts.curation.worker.cascade.
        fetch_pending_multi_project`, which calls :meth:`plan_cycle` for
        the initial, equal quota split and then re-fetches leftover
        capacity from projects whose first-pass fetch came back full.

        Non-idle-poll-due projects are still included in the plan (so
        callers can still bump in-flight caps consistently) but with
        ``should_poll=False`` and ``quota=0``.
        """
        now = time.monotonic() if now is None else now
        del backlog_hint  # see docstring -- kept in the signature for API stability
        n_active = max(1, len(projects))
        in_flight_cap = max(1, math.ceil(queue_max / n_active))

        # Rotation: order projects starting at _rotation_start, advance
        # by one project per cycle for cross-cycle fairness.
        if projects:
            start = self._rotation_start % len(projects)
            ordered = projects[start:] + projects[:start]
            self._rotation_start = (self._rotation_start + 1) % len(projects)
        else:
            ordered = []

        # Which projects are "due" this cycle (idle ones respect backoff;
        # a project with no prior backlog info is treated as due so a
        # brand-new project isn't starved on its first cycle).
        due = [p for p in ordered if self._state_for(p.slug).due(now)]

        quota = max(1, fetch_n // max(1, len(due))) if due else 0
        quotas: dict[str, int] = {p.slug: quota for p in due}

        plans: list[CyclePlan] = []
        for p in ordered:
            should_poll = p in due
            plans.append(
                CyclePlan(
                    slug=p.slug,
                    record=p,
                    quota=quotas.get(p.slug, 0) if should_poll else 0,
                    in_flight_cap=in_flight_cap,
                    should_poll=should_poll,
                )
            )
        return plans


def discover_pollable_projects(all_active: list[ProjectRecord]) -> list[ProjectRecord]:
    """``registry.active_projects()`` filtered to unpaused ones."""
    return [p for p in all_active if not is_project_paused(p)]


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
    exclude_ids: list | None = None,
    backlog_hint: dict[str, int] | None = None,
) -> list:
    """The multi-project discovery + deficit-round-robin producer step
    (projects_plan.md §5.1). Directly unit-testable: it takes a
    ``registry`` (anything with ``.active_projects()``) and a
    :class:`FairnessScheduler`, plans the cycle's initial equal quota
    split, fetches each due, unpaused project under its own binding,
    then runs a real work-conserving second pass: any project whose
    first-pass fetch came back FULL (it could plausibly use more) gets
    another shot at whatever capacity the round didn't use, one project
    at a time, until ``fetch_n`` is exhausted or no full project
    remains. Returns one flat list of ``_ItemTask`` tagged with
    ``project``.

    ``exclude_ids`` is applied to every project's fetch (in-flight ids
    are a worker-wide set today, not per project, since a crop_id is
    already globally unique across projects by construction of the
    items index).
    """
    from scripts.curation.worker.cascade import _fetch_pending
    from src.config.project_context import bind_project

    active = discover_pollable_projects(registry.active_projects())
    if not active:
        return []
    plans = scheduler.plan_cycle(
        active, fetch_n=fetch_n, queue_max=queue_max, backlog_hint=backlog_hint
    )
    tasks: list = []
    total_fetched = 0
    # full_projects: a project that used its FULL first-pass quota is a
    # candidate for leftover capacity (an empty/partial answer means it
    # has nothing more to give right now).
    full_projects: list = []
    for plan in plans:
        if not plan.should_poll or plan.quota <= 0:
            continue
        with bind_project(plan.record):
            fetched = await _fetch_pending(
                opensearch,
                batch_size=plan.quota,
                exclude_ids=exclude_ids,
                project=plan.record,
            )
        scheduler.record_fetch_result(plan.slug, len(fetched))
        tasks.extend(fetched)
        total_fetched += len(fetched)
        if len(fetched) >= plan.quota:
            full_projects.append(plan.record)

    # Second pass: work-conserving leftover redistribution within this
    # same cycle, using this cycle's REAL fetch results (not a guess).
    remaining = max(0, fetch_n - total_fetched)
    while remaining > 0 and full_projects:
        record = full_projects.pop(0)
        with bind_project(record):
            extra = await _fetch_pending(
                opensearch,
                batch_size=remaining,
                exclude_ids=exclude_ids,
                project=record,
            )
        scheduler.record_fetch_result(record.slug, len(extra))
        tasks.extend(extra)
        remaining -= len(extra)
        if len(extra) > 0 and remaining > 0:
            # Still hungry -- this project may have even more; give it
            # another turn once every other full project has had one.
            full_projects.append(record)
    return tasks
