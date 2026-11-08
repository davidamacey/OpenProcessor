"""Start and read model-export jobs (API side).

The API image carries no TensorRT, so ``POST /models/upload`` and ``POST /models/{name}/export``
start a ``model_export`` job (kind in ``src/config/job_kinds/gpu_export.py``, exclusive per Triton
model name) that the ``gpu_export`` consumer runs in the export image. The job row is the task:
its id is the ``task_id`` the status routes take, its progress snapshot and result carry what
``ExportTaskStatus`` shows. The route answers 503 while no consumer listens on the queue.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from src.clients import task_queue
from src.config.queue import TaskQueues
from src.db.repos import engine_builds, jobs as jobs_repo
from src.db.repos.jobs_listing import JobFilter, get_one, list_page
from src.db.session import async_session_factory, deployment_tx
from src.schemas.models import ExportStatus, ExportTaskStatus
from src.services.jobs.service import StartResult, rejection, start_job
from src.services.model_export import ExportWorkerUnavailableError, StagedUpload


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

KIND = 'model_export'
WORKER_SERVICE = 'model-export-worker'
INSPECT_TIMEOUT_S = 1.0
LIST_LIMIT = 200
_STAGES = {
    'validating': ExportStatus.VALIDATING,
    'exporting': ExportStatus.EXPORTING,
    'loading': ExportStatus.LOADING,
}
_FAMILY = 'yolo'


# --- is anything listening on the queue ----------------------------------------------------


def _consumer_listens() -> bool:
    replies = task_queue.producer_app().control.inspect(timeout=INSPECT_TIMEOUT_S).active_queues()
    return any(
        queue.get('name') == TaskQueues.GPU_EXPORT
        for queues in (replies or {}).values()
        for queue in queues
    )


async def require_export_consumer() -> None:
    """Refuse (503 at the route) when no worker consumes ``gpu_export``: nothing would run the job."""
    if not await asyncio.to_thread(_consumer_listens):
        raise ExportWorkerUnavailableError(
            f'The model export worker is not running: no consumer listens on {TaskQueues.GPU_EXPORT}. '
            'Engine builds run in the export image, not in the API. Start it with: '
            f'docker compose --profile export up -d {WORKER_SERVICE}'
        )


# --- start ---------------------------------------------------------------------------------


async def ensure_not_building(session: AsyncSession, triton_name: str) -> None:
    """Answer the 409 before a route replaces the weights a running build is reading."""
    async with deployment_tx(session, 'model export: is this model being built'):
        active = await session.run_sync(
            lambda sync: jobs_repo.active_job_id(sync, None, KIND, triton_name)
        )
    if active is not None:
        raise rejection(
            409,
            'job_busy',
            f'another {KIND} job is already active for {triton_name}',
            active_job_id=str(active),
        )


async def start_export(
    *,
    triton_name: str,
    pt_file: Path,
    max_batch: int,
    formats: Sequence[str],
    normalize_boxes: bool,
    auto_load: bool,
    model_info: Mapping[str, Any],
    new_version: bool,
    requested_by: str = 'api',
    staged: StagedUpload | None = None,
) -> StartResult:
    """Queue one export: the job and, in its transaction, the ``models`` version it builds.

    ``new_version`` is an upload (a new version of ``triton_name``); a re-export reuses the
    latest version. 409 ``job_busy`` while another export of the same name is active.
    ``staged`` (an upload) is placed as the stored weights in the job's transaction, after the
    insert accepted the job and before anything is sent: a refused start replaces nothing.
    """
    params = {
        'pt_file': str(pt_file),
        'triton_name': triton_name,
        'model_name': pt_file.stem,
        'max_batch': max_batch,
        'formats': list(formats),
        'normalize_boxes': normalize_boxes,
        'auto_load': auto_load,
        'head_kind': model_info.get('head_kind'),
        'num_classes': model_info.get('num_classes'),
        'class_names': list(model_info.get('class_names') or []),
    }

    async def model_row(session: AsyncSession, _job: jobs_repo.JobRow) -> None:
        if staged is not None:
            await asyncio.to_thread(staged.place)

        def write(sync: Any) -> None:
            existing = (
                None if new_version else engine_builds.latest_model_version(sync, triton_name)
            )
            if existing is None:
                engine_builds.add_model_version(
                    sync,
                    name=triton_name,
                    labels=model_info.get('class_names') or [],
                    family=_FAMILY,
                    artifact_storage_key=f'pytorch_models/{pt_file.name}',
                    created_by=requested_by,
                )

        await session.run_sync(write)

    async with async_session_factory()() as session:
        return await start_job(
            session, KIND, None, params=params, requested_by=requested_by, with_rows=model_row
        )


# --- read ----------------------------------------------------------------------------------


def _percent(progress: Mapping[str, Any]) -> float:
    done, total = progress.get('done'), progress.get('total')
    if not total or done is None:
        return 0.0
    return float(max(0.0, min(100.0, 100.0 * done / total)))


def _status_of(row: jobs_repo.JobRow) -> ExportStatus:
    if row.status in {'queued', 'interrupted', 'paused'}:
        return ExportStatus.PENDING
    if row.status in {'running', 'cancelling'}:
        return _STAGES.get(str((row.progress or {}).get('stage')), ExportStatus.VALIDATING)
    if row.status == 'failed':
        return ExportStatus.FAILED
    if row.status == 'cancelled':
        return ExportStatus.CANCELLED
    return ExportStatus.COMPLETED


def task_status(row: jobs_repo.JobRow) -> ExportTaskStatus:
    """The ``ExportTaskStatus`` of one ``model_export`` job row."""
    params, progress, result = row.params or {}, row.progress or {}, row.result or {}
    counters = progress.get('counters') or {}
    status = _status_of(row)
    if status is ExportStatus.COMPLETED:
        progress_pct, step = 100.0, 'Complete'
    else:
        progress_pct, step = _percent(progress), str(counters.get('step') or 'Queued for export')
    if status is ExportStatus.FAILED:
        message = f'Export failed: {row.error_message}'
    elif status is ExportStatus.CANCELLED:
        message = 'Export cancelled'
    elif status is ExportStatus.COMPLETED:
        message = f'Model {params.get("triton_name")} exported successfully'
    else:
        message = step
    formats = list(params.get('formats') or [])
    done = status is ExportStatus.COMPLETED
    return ExportTaskStatus(
        task_id=str(row.id),
        model_name=str(params.get('model_name') or params.get('triton_name')),
        triton_name=str(params.get('triton_name')),
        status=status,
        progress=progress_pct,
        current_step=step,
        message=message,
        created_at=row.created_at,
        started_at=row.started_at,
        completed_at=row.finished_at,
        error=row.error_message,
        formats_completed=formats if done else [],
        formats_pending=[] if done else formats,
        num_classes=params.get('num_classes'),
        class_names=params.get('class_names') or None,
        triton_loaded=bool(result.get('triton_loaded', False)),
        export_duration_seconds=result.get('export_duration_seconds'),
        step_times=result.get('step_times') or counters.get('step_times'),
    )


_FILTER = JobFilter(project_id=None, kinds=(KIND,))


async def get_export_task(task_id: str) -> ExportTaskStatus | None:
    try:
        job_id = uuid.UUID(task_id)
    except ValueError:
        return None
    async with async_session_factory()() as session, deployment_tx(session, 'model export status'):
        found = await session.run_sync(lambda sync: get_one(sync, _FILTER, job_id))
    return None if found is None else task_status(found.row)


async def list_export_tasks() -> list[ExportTaskStatus]:
    """The newest ``LIST_LIMIT`` export tasks, newest first."""
    async with async_session_factory()() as session, deployment_tx(session, 'model export list'):
        page = await session.run_sync(lambda sync: list_page(sync, _FILTER, limit=LIST_LIMIT))
    return [task_status(item.row) for item in page]
