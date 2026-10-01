"""Per-project job inventory (docs/design/openprocessor_internal/
projects_plan.md §5.4): what's currently running for one project, across
every job-producing subsystem.

Each source is its own small function so a unit test can fake one at a
time. :func:`running_jobs` is the single entry point P3's delete/archive
busy checks consume (``lifecycle.running_jobs`` adapts this module's
``JobRef`` into the wire-shaped one; see that function).

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
    from pathlib import Path

    from src.config.projects import ProjectRecord


# Non-terminal train run states (jobs.TrainJobStatus.state) -- kept as a
# local literal set rather than importing jobs.TERMINAL-ish constants,
# since 'lost' (a heartbeat-staleness *display* state, not one the
# trainer itself writes) is deliberately excluded here: a lost run's
# trainer process is presumed dead, so it should not block a delete/
# archive the way a genuinely live run does.
_TRAIN_NON_TERMINAL = frozenset({'queued', 'starting', 'running', 'exporting'})


def _iso_or_none(value: object) -> str | None:
    """A job source's raw ``started_at`` -- already an ISO string, a
    numeric epoch (``time.time()``, as autolabel's ``state.json``
    stores it), or absent -- normalized to an ISO string or ``None``
    (P3F m5: never a fabricated ``''``)."""
    if isinstance(value, str) and value:
        return value
    # A 0.0 (or negative) epoch is these dataclasses' own "not actually
    # set yet" sentinel default (e.g. probe/scores/select/viz _JobState),
    # not a real 1970 start time -- surface it as unknown, not a
    # misleadingly precise fake timestamp.
    if isinstance(value, int | float) and value > 0:
        import datetime as _dt

        try:
            return _dt.datetime.fromtimestamp(value, _dt.UTC).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    return None


@dataclass(frozen=True)
class JobRef:
    """One busy job, named well enough for a delete/archive busy-check
    error message ("cars has 2 running jobs: train:20260101_x,
    bakeoff:20260102_y")."""

    kind: str
    job_id: str
    # P3F m5: a real ISO timestamp when the job source actually records a
    # start time, else None -- never an always-empty-string filler.
    started_at: str | None = None
    # P3F pass-3 m-b: a genuine human-readable label when the job
    # source actually records one (e.g. a train run's own submitted
    # ``mlflow_run_name``), else None. ``lifecycle.running_jobs`` falls
    # back to ``job_id`` ONLY when this is None/empty -- never silently
    # pretends the internal job id IS a human label.
    label: str | None = None


def _train_job_label(jobs_dir: Path, job_id: str) -> str | None:
    """The train run's own ``mlflow_run_name`` (P3F pass-3 m-b), read
    from its ``<job_id>.job.json`` submit-time spec -- a genuine human
    label when the submitter set one. ``None`` when no ``job.json`` is
    readable, or it never set a name, so the caller falls back to the
    internal ``job_id`` instead."""
    import json

    spec_path = jobs_dir / f'{job_id}.job.json'
    try:
        payload = json.loads(spec_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    # n-f: a `job.json` can be valid JSON but not an object (e.g. a bare
    # list) -- from hand-editing, an older/different writer, or
    # corruption. Treat that the same as missing/unreadable (fall back to
    # the job id) instead of crashing the busy preflight with
    # AttributeError from `.get` on a non-dict.
    if not isinstance(payload, dict):
        return None
    name = payload.get('mlflow_run_name')
    return name if isinstance(name, str) and name else None


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
            started_at = _iso_or_none(payload.get('started_at'))
            label = _train_job_label(jobs_dir, job_id)
            out.append(JobRef(kind='train', job_id=job_id, started_at=started_at, label=label))
    return out


def _bakeoff_jobs(record: ProjectRecord) -> list[JobRef]:
    """Queued/running bake-off jobs in the project's own
    ``bakeoff_jobs_dir`` -- a job file present (not yet moved to
    ``done/`` by the evaluator, see
    ``scripts.curation.bakeoff.bakeoff_runner``) means it's still
    pending or in flight.

    ``started_at`` comes from the companion evaluator-written
    ``<bakeoff_jobs_dir>/out/<job_id>/status.json`` (``BakeoffStatus``,
    P3F m5) when it exists yet -- a still-``queued`` job (no ``status.json``
    written) reports ``None``, which is honest: it has not started."""
    import json

    from src.services.training.run_retention import bakeoff_out_dir

    jobs_dir = record.resources.bakeoff_jobs_dir
    if not jobs_dir.is_dir():
        return []
    out_root = bakeoff_out_dir(jobs_dir)
    out: list[JobRef] = []
    for p in sorted(jobs_dir.glob('*.job.json')):
        job_id = p.name[: -len('.job.json')]
        started_at = None
        status_file = out_root / job_id / 'status.json'
        if status_file.is_file():
            try:
                status_payload = json.loads(status_file.read_text(encoding='utf-8'))
                started_at = _iso_or_none(status_payload.get('started_at'))
            except (OSError, ValueError):
                started_at = None
        out.append(JobRef(kind='bakeoff', job_id=job_id, started_at=started_at))
    return out


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
    return [
        JobRef(
            kind='autolabel',
            job_id=str(payload.get('job_id') or 'autolabel'),
            started_at=_iso_or_none(payload.get('started_at')),
        )
    ]


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
                state = module._read_state()
                job_id = state.job_id
                started_at = _iso_or_none(getattr(state, 'started_at', None))
                out.append(JobRef(kind=kind, job_id=str(job_id or kind), started_at=started_at))
    return out


def _reprocess_jobs(record: ProjectRecord) -> list[JobRef]:
    """Live reprocess jobs (``src.services.curation.reprocess_job``): one
    state dir per job under the project's reprocess jobs dir, resolved from
    the bound project, so this binds ``record`` just for the read."""
    from src.config.project_context import bind_project
    from src.services.curation import reprocess_job

    with bind_project(record, read_only=True):
        return [
            JobRef(kind='reprocess', job_id=job_id, started_at=_iso_or_none(started_at))
            for job_id, started_at in reprocess_job.running_job_ids()
        ]


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
        # started_at stays None: this is a liveness heartbeat
        # (updated_at), never a job start time.
        out.append(JobRef(kind='detection_worker', job_id=host))
    return out


def _dataset_import_jobs(record: ProjectRecord) -> list[JobRef]:
    """Live dataset imports (``src.services.curation.dataset_import``): one
    state dir per import under the project's imports dir, resolved from the
    bound project, so this binds ``record`` just for the read. Live means an
    active status with a fresh heartbeat (a dead worker's import is not busy:
    startup repair marks it ``interrupted``)."""
    from src.config.project_context import bind_project
    from src.services.curation.dataset_import.store import running_import_ids

    with bind_project(record, read_only=True):
        return [
            JobRef(kind='dataset_import', job_id=import_id, started_at=_iso_or_none(started_at))
            for import_id, started_at in running_import_ids()
        ]


def _combine_jobs(record: ProjectRecord) -> list[JobRef]:
    """Live combine jobs that read or write this project (as a source or as
    the target being built); a combine is global, so its jobs dir is not
    nested under a project."""
    from src.services.projects.combine.store import running_jobs_for

    return [
        JobRef(kind='combine', job_id=job_id, started_at=_iso_or_none(started_at))
        for job_id, started_at in running_jobs_for(record.slug)
    ]


def running_jobs(record: ProjectRecord) -> list[JobRef]:
    """Every busy job for ``record``'s project, across every source.

    Consumed by P3's delete/archive busy checks
    (``lifecycle.running_jobs`` adapts these into the wire-shaped
    ``JobRef``). Each source function above can be faked independently
    in a test.
    """
    jobs: list[JobRef] = []
    jobs.extend(_train_jobs(record))
    jobs.extend(_bakeoff_jobs(record))
    jobs.extend(_autolabel_jobs(record))
    jobs.extend(_export_jobs(record))
    jobs.extend(_state_file_jobs(record))
    jobs.extend(_reprocess_jobs(record))
    jobs.extend(_detection_worker_inflight(record))
    jobs.extend(_dataset_import_jobs(record))
    jobs.extend(_combine_jobs(record))
    return jobs


__all__ = ['JobRef', 'running_jobs']
