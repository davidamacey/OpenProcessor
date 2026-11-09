"""When the periodic cluster refresh starts a ``cluster_run``.

Scheduling only: the run itself is the ``cluster_run`` job. The beat tick asks this module, per
project and per tick, whether to start one now. Pure logic with an injected clock so it is tested
without a stack.

* A run is requested once the item pool has grown by ``threshold`` past the pool of the newest
  finished partition (``built``), or once a never-clustered project has any item. An empty project
  requests nothing.
* It never starts while an assist loop runs on the project (the loop works on a pinned partition;
  a run would move its members). The request is kept and runs after.
* It starts when ingest is quiet: the pool did not change since the previous tick, no region work
  is unfinished and no image was ingested within the settle window (a slow ingest batch leaves the
  count flat between ticks). While ingest runs, new items are already assigned to the serving
  partition's prototypes at ingest time.
* ``max_deferral_s`` bounds the wait so a long ingest never starves clustering.
* Requests coalesce: one pending clock per project, however many ticks cross the threshold. A
  request raised while a run is live is kept and runs once after it.
* A run that ended badly (failed, cancelled, interrupted, or lost) leaves no partition, so the
  growth still stands and the request is retried after ``retry_backoff_s``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


ACTIVE_JOB_STATUSES = frozenset({'queued', 'running', 'paused', 'cancelling'})


class Action(Enum):
    NONE = 'none'
    DEFER = 'defer'
    START = 'start'


@dataclass(frozen=True)
class Policy:
    threshold: float
    max_deferral_s: float
    retry_backoff_s: float


@dataclass(frozen=True)
class Observation:
    count: int  # items in the clustering pool now
    built: int  # pool size of the newest finished partition, 0 when there is none
    unfinished: int  # items still waiting for the region stage
    tracked_status: str | None  # status of the run this schedule started, if any
    busy: bool  # any cluster_run of the project is live (not only ours)
    loop_live: bool  # an assist loop is running: nothing may re-cluster under it
    recent_ingest: bool  # any image ingested within the settle window


@dataclass
class ProjectSchedule:
    """The clock a tick carries to the next one (``cluster_refresh_clock``); wall-clock seconds."""

    prev_count: int | None = None
    pending_since: float | None = None
    retry_not_before: float = 0.0
    job_id: str | None = None

    def on_started(self, job_id: str) -> None:
        self.job_id = job_id
        self.pending_since = None


def step(s: ProjectSchedule, o: Observation, now: float, p: Policy) -> Action:
    quiet = (
        o.unfinished == 0
        and not o.recent_ingest
        and (s.prev_count is None or o.count == s.prev_count)
    )
    s.prev_count = o.count

    in_flight = False
    if s.job_id is not None:
        if o.tracked_status in ACTIVE_JOB_STATUSES:
            in_flight = True
        elif o.tracked_status == 'completed':
            s.job_id = None
        else:
            s.job_id = None
            s.retry_not_before = now + p.retry_backoff_s
            return Action.NONE

    # built == 0 means never clustered; an empty project has nothing to cluster.
    crossed = o.count > 0 and (o.built == 0 or o.count - o.built >= p.threshold)
    if crossed and s.pending_since is None:
        s.pending_since = now
    if not crossed and not in_flight:
        s.pending_since = None
    if s.pending_since is None or in_flight or o.busy or o.loop_live or now < s.retry_not_before:
        return Action.NONE
    if quiet or now - s.pending_since >= p.max_deferral_s:
        return Action.START
    return Action.DEFER
