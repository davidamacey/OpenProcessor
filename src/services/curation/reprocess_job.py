"""The file-backed reprocess job (``detect`` / ``embed`` over many images).

State lives under ``<OP_REPROCESS_JOBS_DIR>/projects/<slug>/<job_id>/`` on
the shared ``/jobs`` volume (:class:`~src.services.curation.file_job.FileJob`):
every API worker process sees the same ``state.json`` / heartbeat / cancel
flag, so status and cancel work whichever process a request lands on, and
one job per project runs at a time (``start.lock`` + a live-job check).

``request.json`` is everything the run needs: the resolved image ids, the
scopes, and the index names the request was planned against.
:func:`run_job` reads only that and the dependencies it is handed, so a
process that never saw the API call can consume it; it refuses to run if
the bound project's indexes are not the ones the request was planned for.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.config.project_context import current_project, project_jobs_dir
from src.core.logging import get_logger
from src.services.curation.file_job import FileJob, heartbeat_ticker
from src.services.curation.job_lock import exclusive_start_lock
from src.services.curation.reprocess_images import process_images
from src.services.curation.reprocess_models import ReprocessJobInfo, ReprocessScope


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.ingest import CurationIngestService

logger = get_logger(__name__)

ACTIVE = frozenset({'queued', 'running'})

_tasks: dict[str, asyncio.Task[None]] = {}


class ReprocessBusyError(Exception):
    """A reprocess job is already running for this project."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f'reprocess job {job_id} is already running')
        self.job_id = job_id


def jobs_root() -> Path:
    """The bound project's reprocess jobs dir (resolved per call so tests
    can ``monkeypatch.setenv``)."""
    return project_jobs_dir(Path(os.environ.get('OP_REPROCESS_JOBS_DIR', '/jobs/reprocess')))


def _job(job_id: str) -> FileJob:
    # job ids are minted here; never let a path segment through.
    if not job_id or '/' in job_id or '\\' in job_id or job_id.startswith('.'):
        raise KeyError(job_id)
    return FileJob(jobs_root() / job_id)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _job_dirs() -> list[Path]:
    root = jobs_root()
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def running_job_ids() -> list[tuple[str, str | None]]:
    """``[(job_id, started_at)]`` of live jobs in the bound project."""
    out = []
    for path in _job_dirs():
        job = FileJob(path)
        if job.is_live(ACTIVE):
            out.append((path.name, job.read().get('started_at')))
    return out


def reconcile_orphaned_jobs() -> bool:
    """Startup repair for the bound project: a job left active by a dead
    process becomes ``interrupted``."""
    repaired = False
    for path in _job_dirs():
        repaired |= FileJob(path).reconcile(active_statuses=ACTIVE, error_prefix='reprocess job')
    return repaired


def read_job(job_id: str) -> ReprocessJobInfo | None:
    try:
        job = _job(job_id)
    except KeyError:
        return None
    state = job.read()
    if not state:
        return None
    state = job.repair_if_stale(ACTIVE, error_prefix='reprocess job')
    for key in ('started_at', 'updated_at', 'finished_at'):
        # Startup reconciliation stamps ``finished_at`` as an epoch number.
        if isinstance(state.get(key), int | float):
            state[key] = datetime.fromtimestamp(state[key], UTC).isoformat()
    return ReprocessJobInfo(
        job_id=job_id,
        poll_after_s=2 if state.get('status') in ACTIVE else None,
        **{k: v for k, v in state.items() if k in ReprocessJobInfo.model_fields and k != 'job_id'},
    )


def create_job(
    *, request: dict[str, Any], scopes: list[ReprocessScope], image_ids: list[str]
) -> FileJob:
    """Claim the per-project singleton and persist ``request.json`` plus a
    ``queued`` state. Raises :class:`ReprocessBusyError` when a job is
    already live (or another process is mid-claim). Nothing runs yet."""
    root = jobs_root()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_start_lock(root / 'start.lock') as acquired:
        live = running_job_ids()
        if not acquired or live:
            raise ReprocessBusyError(live[0][0] if live else 'starting')
        job_id = f'rp_{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}'
        job = _job(job_id)
        job.directory.mkdir(parents=True)
        cfg = get_curation_config()
        (job.directory / 'request.json').write_text(
            json.dumps(
                {
                    'request': request,
                    'scopes': scopes,
                    'image_ids': image_ids,
                    'project': current_project().record.slug,
                    'items_index': cfg.items_index,
                    'images_index': cfg.images_index,
                }
            ),
            encoding='utf-8',
        )
        job.clear_signals()
        job.write(
            {
                'status': 'queued',
                'scopes': scopes,
                'images_total': len(image_ids),
                'images_done': 0,
                'images_failed': 0,
                'started_at': _now(),
                'updated_at': _now(),
            }
        )
        job.touch_heartbeat()
    return job


def start_job(
    opensearch: AsyncOpenSearch,
    service: CurationIngestService,
    *,
    request: dict[str, Any],
    scopes: list[ReprocessScope],
    image_ids: list[str],
) -> str:
    """:func:`create_job`, then schedule :func:`run_job` in this process."""
    job = create_job(request=request, scopes=scopes, image_ids=image_ids)
    _tasks[job.directory.name] = asyncio.create_task(run_job(job, opensearch, service))
    return job.directory.name


def cancel_job(job_id: str) -> bool:
    """Touch the cancel flag of a live job; ``False`` when there is none."""
    try:
        job = _job(job_id)
    except KeyError:
        return False
    if not job.is_live(ACTIVE):
        return False
    job.request_cancel()
    return True


async def run_job(
    job: FileJob, opensearch: AsyncOpenSearch, service: CurationIngestService
) -> None:
    """Execute the job persisted at ``job.directory``. Needs nothing but the
    files there and the dependencies passed in."""
    try:
        payload = json.loads((job.directory / 'request.json').read_text(encoding='utf-8'))
        cfg = get_curation_config()
        if (payload['items_index'], payload['images_index']) != (cfg.items_index, cfg.images_index):
            raise RuntimeError(
                "the job was planned for other indexes than the bound project's; refusing to run"
            )
    except Exception as exc:
        job.update(status='failed', error=str(exc), finished_at=_now(), updated_at=_now())
        return
    job.update(status='running', updated_at=_now())
    ticker = asyncio.create_task(heartbeat_ticker(job))

    def _progress(done: int, failed: int) -> None:
        job.update(images_done=done, images_failed=failed, updated_at=_now())
        job.touch_heartbeat()

    try:
        results, cancelled = await process_images(
            opensearch,
            service,
            scopes=payload['scopes'],
            image_ids=payload['image_ids'],
            should_cancel=job.cancel_requested,
            on_progress=_progress,
        )
        failed = any(r.failed or r.not_found for r in results)
        status = 'cancelled' if cancelled else 'completed_with_errors' if failed else 'completed'
        job.update(
            status=status,
            results=[r.model_dump() for r in results],
            finished_at=_now(),
            updated_at=_now(),
        )
    except asyncio.CancelledError:
        job.update(status='cancelled', finished_at=_now(), updated_at=_now())
        raise
    except Exception as exc:
        logger.error('reprocess_job_failed', job=job.directory.name, error=str(exc))
        job.update(status='failed', error=str(exc), finished_at=_now(), updated_at=_now())
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker


__all__ = [
    'ACTIVE',
    'ReprocessBusyError',
    'cancel_job',
    'create_job',
    'jobs_root',
    'read_job',
    'reconcile_orphaned_jobs',
    'run_job',
    'running_job_ids',
    'start_job',
]
