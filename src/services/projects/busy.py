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

import importlib
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


def _autolabel_jobs(record: ProjectRecord) -> list[JobRef]:
    """Auto-label worker state (``src.services.curation.autolabel.job``).

    ``autolabel_dir`` is a PROJECT_SCOPED_FIELDS entry
    (``resources_for_new`` (used for every project, ``default`` included)); the worker/job
    module resolves it from the *bound* project context, so reading
    another project's state without binding means reading its
    ``state.json`` directly here rather than calling ``get_state()``
    (which also runs stale-heartbeat repair as a side effect -- not this
    read-only inventory's job). ``'running'`` is the only busy status
    (see ``_JobState.status``); a missing/unreadable file means idle.
    """
    import json

    state_file = record.resources.autolabel_dir / 'state.json'
    if not state_file.is_file():
        return []
    try:
        payload = json.loads(state_file.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    if payload.get('status') != 'running':
        return []
    return [JobRef(kind='autolabel', job_id=str(payload.get('job_id') or 'autolabel'))]


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


# The in-process curation job runners that keep one per-project
# ``state.json`` + ``heartbeat`` pair under ``<OP_*_JOBS_DIR>/projects/<slug>``
# (kind -> module). Each module's own ``_is_busy()`` is the busy rule:
# ``status == 'running'`` and a heartbeat no older than its
# ``_HEARTBEAT_STALE_S`` (no heartbeat yet = just started = busy).
_STATE_JOB_MODULES: dict[str, str] = {
    'probe': 'src.services.curation.probe_job',
    'scores': 'src.services.curation.item_scores.job',
    'select': 'src.services.curation.selection.job',
    'viz': 'src.services.curation.embedding_viz',
}


def _state_file_jobs(record: ProjectRecord) -> list[JobRef]:
    """Running probe, item-scores, selection and viz jobs.

    Each module resolves its dir from the *bound* project, so this binds
    ``record`` just for the read (nesting restores the caller's binding)
    and reuses the module's own ``_is_busy()`` / ``_read_state()`` rather
    than re-deriving the path or the staleness rule.
    """
    from src.config.project_context import bind_project

    out: list[JobRef] = []
    with bind_project(record, read_only=True):
        for kind, module_name in _STATE_JOB_MODULES.items():
            module = importlib.import_module(module_name)
            if module._is_busy():
                job_id = module._read_state().job_id
                out.append(JobRef(kind=kind, job_id=str(job_id or kind)))
    return out


def _detection_worker_inflight(record: ProjectRecord) -> list[JobRef]:
    """The detection worker's per-project ``inflight`` count.

    ``scripts/curation/worker/fairness.py``'s ``write_liveness`` drops one
    ``runtime_detection_worker_<host>.json`` file per host under this
    project's own ``project_state_dir`` every cycle (§5.1) -- a real,
    file-based liveness doc, not the OpenSearch ``runtime:*`` doc the
    plan sketched (no such mechanism exists on this branch; see that
    module's docstring). One busy entry per host actually mid-batch
    (``inflight > 0``), keyed by hostname so a delete/archive busy-check
    error message can name which host is still working the project.

    A file whose ``updated_at`` is older than the worker heartbeat window
    (``worker_liveness.DEFAULT_MAX_AGE_S``, the same max age the worker
    healthchecks use) belongs to a worker that died mid-batch: it is not
    busy, or the project could never be deleted. A file with no numeric
    ``updated_at`` (``write_liveness`` always stamps one) counts as stale.
    """
    import json
    import time

    from src.services.curation.worker_liveness import DEFAULT_MAX_AGE_S

    state_dir = record.resources.project_state_dir
    if not state_dir.is_dir():
        return []
    out: list[JobRef] = []
    for liveness_file in sorted(state_dir.glob('runtime_detection_worker_*.json')):
        try:
            payload = json.loads(liveness_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        inflight = int(payload.get('inflight') or 0)
        if inflight <= 0:
            continue
        updated_at = payload.get('updated_at')
        if not isinstance(updated_at, int | float) or (
            time.time() - updated_at > DEFAULT_MAX_AGE_S
        ):
            continue
        host = str(payload.get('host') or liveness_file.stem)
        out.append(JobRef(kind='detection_worker', job_id=host))
    return out


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
    jobs.extend(_state_file_jobs(record))
    jobs.extend(_detection_worker_inflight(record))
    return jobs


__all__ = ['JobRef', 'running_jobs']
