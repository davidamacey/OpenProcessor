"""When the cluster-refresh daemon starts a full cluster retrain.

Scheduling only: the retrain itself is the auto-label job, unchanged (same
method, parameters, seed and pool). The daemon asks this module, per project
and per poll, whether to start it now. Pure logic with an injected clock so
it is tested without a stack.

* A retrain is requested once the item count has grown by ``threshold``
  since the last one dispatched (or never trained).
* It starts when ingest is quiet: the item count did not change since the
  previous poll and no region work is unfinished. While ingest runs, new
  items are already assigned to the persisted centroids at ingest time.
* ``max_deferral_s`` bounds the wait so a long ingest never starves
  clustering.
* Requests coalesce: one pending flag per project, however many polls cross
  the threshold. A request raised while a job runs is kept and runs once
  after it.
* A job that ended badly (failed, cancelled, interrupted by the stale
  repair, or lost) hands its trained mark back, so the request is raised
  again and retried after ``retry_backoff_s``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


ACTIVE_JOB_STATUSES = frozenset({'queued', 'running'})


class Action(Enum):
    NONE = 'none'
    DEFER = 'defer'
    START = 'start'


@dataclass(frozen=True)
class Policy:
    threshold: int
    max_deferral_s: float
    retry_backoff_s: float


@dataclass(frozen=True)
class Observation:
    count: int
    unfinished: int  # items still waiting for the region stage
    tracked_status: str | None  # status of the job this daemon started, if any
    busy: bool  # any auto-label job (not only ours) is queued or running


@dataclass
class ProjectSchedule:
    last_trained_count: int = 0  # item count when the last retrain was dispatched
    trained_before: int | None = None  # the mark to restore if that job does not finish
    prev_count: int | None = None
    pending_since: float | None = None
    retry_not_before: float = 0.0
    job_id: str | None = None

    def on_started(self, count: int, job_id: str | None, *, previous_trained: int) -> None:
        self.trained_before = previous_trained
        self.last_trained_count = count
        self.job_id = job_id
        self.pending_since = None

    def on_cancelled(self) -> None:
        """The daemon cancelled its job (project paused): the work was not done."""
        self._restore()

    def _restore(self) -> None:
        if self.trained_before is not None:
            self.last_trained_count = self.trained_before
        self.trained_before = None
        self.job_id = None


@dataclass
class ScheduleBook:
    """One schedule per project slug; projects never share state."""

    _by_slug: dict[str, ProjectSchedule] = field(default_factory=dict)

    def get(self, slug: str) -> ProjectSchedule:
        return self._by_slug.setdefault(slug, ProjectSchedule())

    def forget(self, slug: str) -> None:
        self._by_slug.pop(slug, None)

    def slugs(self) -> list[str]:
        return list(self._by_slug)


def step(s: ProjectSchedule, o: Observation, now: float, p: Policy) -> Action:
    quiet = o.unfinished == 0 and (s.prev_count is None or o.count == s.prev_count)
    s.prev_count = o.count

    in_flight = False
    if s.job_id is not None:
        if o.tracked_status in ACTIVE_JOB_STATUSES:
            in_flight = True
        elif o.tracked_status == 'completed':
            s.job_id = None
            s.trained_before = None
        else:
            s._restore()
            s.retry_not_before = now + p.retry_backoff_s
            return Action.NONE

    crossed = s.last_trained_count == 0 or o.count - s.last_trained_count >= p.threshold
    if crossed and s.pending_since is None:
        s.pending_since = now
    if s.pending_since is None or in_flight or o.busy or now < s.retry_not_before:
        return Action.NONE
    if quiet or now - s.pending_since >= p.max_deferral_s:
        return Action.START
    return Action.DEFER
