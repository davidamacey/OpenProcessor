"""The ``training_runs`` rows: one per ``train`` job, read together with the job's state.

A run's live state (status, epoch progress, heartbeat) is the ``jobs`` row; what the run is and
what it produced (spec, class names, artifact prefix, manifest) is the ``training_runs`` row.
Every function takes the caller's sync ``Session`` inside a scoped transaction and never commits;
an async caller wraps one in ``AsyncSession.run_sync``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select, text, update

from src.db.models.jobs import Job
from src.db.models.training import TrainingRun
from src.db.repos import job_events, jobs as jobs_repo
from src.domain import job_states as js


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Sequence

    from sqlalchemy.orm import Session

TRAIN_KIND = 'train'


@dataclass(frozen=True)
class RunRow:
    """A run with the state of its job."""

    job_id: uuid.UUID
    project_id: uuid.UUID
    spec: dict[str, Any]
    class_names: list[str]
    family: str
    base_model: str
    artifact_prefix: str
    manifest: dict[str, Any] | None
    mlflow_run_id: str | None
    gpu_uuids: list[str]
    status: str
    params: dict[str, Any]
    progress: dict[str, Any]
    error_code: str | None
    error_message: str | None
    created_at: dt.datetime
    started_at: dt.datetime | None
    finished_at: dt.datetime | None
    heartbeat_at: dt.datetime | None


_COLUMNS = (
    TrainingRun.job_id,
    TrainingRun.project_id,
    TrainingRun.spec,
    TrainingRun.class_names,
    TrainingRun.family,
    TrainingRun.base_model,
    TrainingRun.artifact_prefix,
    TrainingRun.manifest,
    TrainingRun.mlflow_run_id,
    TrainingRun.gpu_uuids,
    Job.status,
    Job.params,
    Job.progress,
    Job.error_code,
    Job.error_message,
    Job.created_at,
    Job.started_at,
    Job.finished_at,
    Job.heartbeat_at,
)


def _select() -> Any:
    return select(*_COLUMNS).join(Job, Job.id == TrainingRun.job_id)


def _to_row(mapping: Any) -> RunRow:
    return RunRow(
        job_id=mapping['job_id'],
        project_id=mapping['project_id'],
        spec=dict(mapping['spec']),
        class_names=list(mapping['class_names']),
        family=mapping['family'],
        base_model=mapping['base_model'],
        artifact_prefix=mapping['artifact_prefix'],
        manifest=None if mapping['manifest'] is None else dict(mapping['manifest']),
        mlflow_run_id=mapping['mlflow_run_id'],
        gpu_uuids=list(mapping['gpu_uuids']),
        status=mapping['status'],
        params=dict(mapping['params'] or {}),
        progress=dict(mapping['progress'] or {}),
        error_code=mapping['error_code'],
        error_message=mapping['error_message'],
        created_at=mapping['created_at'],
        started_at=mapping['started_at'],
        finished_at=mapping['finished_at'],
        heartbeat_at=mapping['heartbeat_at'],
    )


def insert_run(
    session: Session,
    *,
    project_id: uuid.UUID,
    job_id: uuid.UUID,
    spec: dict[str, Any],
    class_names: Sequence[str],
    family: str,
    base_model: str,
    artifact_prefix: str,
    gpu_uuids: Sequence[str],
) -> None:
    session.add(
        TrainingRun(
            project_id=project_id,
            job_id=job_id,
            spec=spec,
            class_names=list(class_names),
            family=family,
            base_model=base_model,
            artifact_prefix=artifact_prefix,
            gpu_uuids=list(gpu_uuids),
        )
    )
    session.flush()


def get(session: Session, job_id: uuid.UUID) -> RunRow | None:
    found = session.execute(_select().where(TrainingRun.job_id == job_id)).mappings().first()
    return None if found is None else _to_row(found)


def list_page(session: Session, *, limit: int, offset: int) -> list[RunRow]:
    """Newest first, by the job's creation time."""
    query = _select().order_by(Job.created_at.desc(), Job.id.desc()).limit(limit).offset(offset)
    return [_to_row(m) for m in session.execute(query).mappings()]


def count(session: Session) -> int:
    return int(session.execute(select(func.count()).select_from(TrainingRun)).scalar_one())


def active(session: Session) -> list[RunRow]:
    """Runs whose job is not finished, the running one first, then the queue oldest first."""
    query = (
        _select()
        .where(Job.status.in_(sorted(js.ACTIVE)))
        .order_by((Job.status == js.RUNNING).desc(), Job.created_at.asc(), Job.id.asc())
    )
    return [_to_row(m) for m in session.execute(query).mappings()]


def in_campaign(session: Session, campaign_id: str) -> list[RunRow]:
    """Every run of a campaign, oldest first (the order the campaign queued them)."""
    query = (
        _select()
        .where(Job.params['campaign_id'].astext == campaign_id)
        .order_by(Job.created_at.asc(), Job.id.asc())
    )
    return [_to_row(m) for m in session.execute(query).mappings()]


def set_manifest(
    session: Session, job_id: uuid.UUID, manifest: dict[str, Any], mlflow_run_id: str | None
) -> None:
    session.execute(
        update(TrainingRun)
        .where(TrainingRun.job_id == job_id)
        .values(manifest=manifest, mlflow_run_id=mlflow_run_id, finished_at=func.now())
    )


def set_gpu_uuids(session: Session, job_id: uuid.UUID, gpu_uuids: Sequence[str]) -> None:
    """Record the cards the run holds, by UUID (research X-6)."""
    session.execute(
        update(TrainingRun).where(TrainingRun.job_id == job_id).values(gpu_uuids=list(gpu_uuids))
    )


def stamp_promotion(session: Session, job_id: uuid.UUID, promoted_to: dict[str, Any]) -> bool:
    """Set ``manifest.promoted_to``; ``False`` when the run has no manifest yet."""
    run = session.execute(
        select(TrainingRun).where(TrainingRun.job_id == job_id).with_for_update()
    ).scalar_one_or_none()
    if run is None or run.manifest is None:
        return False
    run.manifest = {**run.manifest, 'promoted_to': promoted_to}
    session.flush()
    return True


def skip_queued(
    session: Session, campaign_id: str, except_job_id: uuid.UUID, code: str, message: str
) -> int:
    """Cancel the campaign's runs that have not started, naming ``code`` as the reason.

    Used when a finished run already met the campaign's ``stop_when``; returns how many.
    """
    skipped = 0
    for run in in_campaign(session, campaign_id):
        if run.job_id == except_job_id or run.status != js.QUEUED:
            continue
        moved = jobs_repo.cancel(session, run.job_id, 'campaign')
        if moved is None or moved.status != js.CANCELLED:
            continue
        session.execute(
            text('UPDATE jobs SET error_code = :code, error_message = :msg WHERE id = :id'),
            {'code': code, 'msg': message, 'id': run.job_id},
        )
        job_events.emit_state(session, moved)
        skipped += 1
    return skipped


def link_model(session: Session, model_id: uuid.UUID, run_job_id: uuid.UUID) -> bool:
    """Point ``models.training_run_id`` at the run of ``run_job_id`` (the promote job's
    ``params.run_job_id``); ``False`` when that run is not in the project or the model
    already names one."""
    linked = session.execute(
        text(
            'UPDATE models SET training_run_id = (SELECT id FROM training_runs WHERE job_id = :job) '
            'WHERE id = :model AND training_run_id IS NULL '
            'AND EXISTS (SELECT 1 FROM training_runs WHERE job_id = :job) RETURNING id'
        ),
        {'job': run_job_id, 'model': model_id},
    )
    return linked.first() is not None
