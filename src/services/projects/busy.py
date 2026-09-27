"""Per-project job inventory (docs/design/openprocessor_internal/
projects_plan.md §5.4): what's currently running for one project, across
every job-producing subsystem.

Each source is its own small function so a unit test can fake one at a
time. :func:`running_jobs` is the single entry point P3 (delete/archive
busy checks) will consume -- wiring it into any lifecycle route is out of
scope here; this module only builds the inventory.

Every source reads *file-based* state scoped to the given project's own
``ProjectResources`` (never the live-bound ``current_project()`` context,
so a caller can build the inventory for a project other than the one
bound to the current request) -- no direct OpenSearch calls, so there is
nothing here for the guarded-client rule to apply to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


# Non-terminal train run states (jobs.TrainJobStatus.state) -- kept as a
# local literal set rather than importing jobs.TERMINAL-ish constants,
# since 'lost' (a heartbeat-staleness *display* state, not one the
# trainer itself writes) is deliberately excluded here: a lost run's
# trainer process is presumed dead, so it should not block a delete/
# archive the way a genuinely live run does.
_TRAIN_NON_TERMINAL = frozenset({'queued', 'starting', 'running', 'exporting'})


@dataclass(frozen=True)
class JobRef:
    """One busy job, named well enough for a delete/archive busy-check
    error message ("cars has 2 running jobs: train:20260101_x,
    bakeoff:20260102_y")."""

    kind: str
    job_id: str


def _train_jobs(record: ProjectRecord) -> list[JobRef]:
    """Non-terminal ``*.status.json`` files in the project's own
    ``train_jobs_dir`` (§5.3 -- already project-scoped)."""
    import json

    jobs_dir = record.resources.train_jobs_dir
    if not jobs_dir.is_dir():
        return []
    out: list[JobRef] = []
    for status_file in sorted(jobs_dir.glob('*.status.json')):
        try:
            payload = json.loads(status_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if payload.get('state') in _TRAIN_NON_TERMINAL:
            job_id = status_file.name[: -len('.status.json')]
            out.append(JobRef(kind='train', job_id=job_id))
    return out


def _bakeoff_jobs(record: ProjectRecord) -> list[JobRef]:
    """Queued/running bake-off jobs in the project's own
    ``bakeoff_jobs_dir`` -- a job file present (not yet moved to
    ``done/`` by the evaluator, see
    ``scripts.curation.bakeoff.bakeoff_runner``) means it's still
    pending or in flight."""
    jobs_dir = record.resources.bakeoff_jobs_dir
    if not jobs_dir.is_dir():
        return []
    return [
        JobRef(kind='bakeoff', job_id=p.name[: -len('.job.json')])
        for p in sorted(jobs_dir.glob('*.job.json'))
    ]


def _autolabel_jobs(record: ProjectRecord) -> list[JobRef]:  # noqa: ARG001 - see docstring
    """Auto-label worker state (``src.services.curation.autolabel.job``).

    TODO(other agent's worker runtime doc): ``autolabel/job.py`` still
    reads/writes a single global ``OP_AUTO_LABEL_STATE_DIR`` rather than
    the bound project's own ``CurationConfig.autolabel_dir`` (a
    PROJECT_SCOPED_FIELDS entry already computed per project by
    ``resources_for_new``/``resources_for_default``) -- making that
    per-project is the auto-label worker's own P2 slice, owned by the
    parallel worker-wave agent, not this file. Until that lands there is
    no per-project auto-label state to read without risking a false
    "project X is busy" read of a different project's run, so this
    source deliberately reports nothing rather than guessing.
    """
    return []


def _export_jobs(record: ProjectRecord) -> list[JobRef]:  # noqa: ARG001 - see docstring
    """Export jobs (``src.routers.curation.export``).

    ``POST /export/yolo`` runs synchronously inside the request (see
    ``export_yolo`` in that module) -- there is no background job file
    or running-state marker to poll, so there is never an in-flight
    export to report here. If export ever grows an async job protocol
    (a ``*.status.json`` under a project-scoped export-jobs dir, mirroring
    train/bake-off), this function is the one to extend.
    """
    return []


def _detection_worker_inflight(record: ProjectRecord) -> list[JobRef]:  # noqa: ARG001
    """The detection worker's per-project ``inflight`` count.

    TODO(other agent's worker runtime doc): the detection worker
    (``scripts/curation/worker/``, owned by the parallel agent) is
    expected to publish a per-project ``runtime:detection_worker:<host>``
    doc with an ``inflight`` count. That doc does not exist yet as of
    this file's authorship; stubbed to ``[]`` rather than guessing at a
    doc id/shape that may still change.
    """
    return []


def running_jobs(record: ProjectRecord) -> list[JobRef]:
    """Every busy job for ``record``'s project, across every source.

    Consumed by P3's delete/archive busy checks (not wired here). Each
    source function above can be faked independently in a test.
    """
    jobs: list[JobRef] = []
    jobs.extend(_train_jobs(record))
    jobs.extend(_bakeoff_jobs(record))
    jobs.extend(_autolabel_jobs(record))
    jobs.extend(_export_jobs(record))
    jobs.extend(_detection_worker_inflight(record))
    return jobs


__all__ = ['JobRef', 'running_jobs']
