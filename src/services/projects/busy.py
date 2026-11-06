"""Is this project busy: one query on the project's active ``jobs`` rows (02b section 2).

A project is busy while a job of its own is ``queued``, ``running``, ``paused`` or
``cancelling`` (``job_states.ACTIVE``), or while a combine run that reads it as a source is. There is
no scan of job state files: every job kind has a row in ``jobs`` (D-75), which is the one inventory
that delete, archive and the combine source check consume (``lifecycle_checks.running_jobs``
adapts :class:`JobRef` into the wire-shaped one).

The detection worker is a stream worker, not a job: its in-flight batch is read from the worker's
own per-project liveness file and still counts, so a delete waits for the last batch to drain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config.project_context import ProjectHasNoIdError
from src.db.engine import get_poll_engine
from src.db.session import sync_deployment_tx
from src.domain import job_states


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


@dataclass(frozen=True)
class JobRef:
    """One busy job, named well enough for a delete/archive busy-check
    error message ("cars has 2 running jobs: train:20260101_x,
    bakeoff:20260102_y")."""

    kind: str
    job_id: str
    # A real ISO timestamp when the job has started, else None -- never an always-empty filler.
    started_at: str | None = None
    # A human-readable label when the job source records one, else None.
    # ``lifecycle_checks.running_jobs`` falls back to ``job_id`` ONLY when this is None/empty.
    label: str | None = None


_ACTIVE_JOBS = text(
    'SELECT j.id, j.kind, j.started_at FROM jobs j WHERE j.status = ANY(:active) AND '
    '(j.project_id = :project OR EXISTS (SELECT 1 FROM combine_sources s '
    'WHERE s.job_id = j.id AND s.source_project_id = :project)) ORDER BY j.created_at, j.id'
)


def _active_job_rows(record: ProjectRecord) -> list[JobRef]:
    """The project's active jobs, including combine runs that read it as a source."""
    if record.id is None:
        raise ProjectHasNoIdError(f"project '{record.slug}' has no database id")
    with Session(get_poll_engine()) as session, sync_deployment_tx(session, 'project busy check'):
        found = session.execute(
            _ACTIVE_JOBS, {'active': sorted(job_states.ACTIVE), 'project': record.id}
        )
        return [
            JobRef(
                kind=row.kind,
                job_id=str(row.id),
                started_at=None if row.started_at is None else row.started_at.isoformat(),
            )
            for row in found
        ]


def _detection_worker_inflight(record: ProjectRecord) -> list[JobRef]:
    """The detection worker's per-project ``inflight`` count.

    ``scripts/curation/worker/fairness.py``'s ``write_liveness`` drops one
    ``runtime_detection_worker_<host>.json`` file per host under this
    project's own ``project_state_dir`` every cycle (§5.1) -- a real,
    file-based liveness doc (see that module's docstring). One busy entry per host actually
    mid-batch (``inflight > 0``), keyed by hostname so a delete/archive busy-check
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


def running_jobs(record: ProjectRecord) -> list[JobRef]:
    """Every busy job for ``record``'s project: its active ``jobs`` rows, and the detection
    worker's in-flight batch."""
    return [*_active_job_rows(record), *_detection_worker_inflight(record)]


__all__ = ['JobRef', 'running_jobs']
