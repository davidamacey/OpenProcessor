"""Training runs: start one (a ``train`` job and its ``training_runs`` row), read, cancel.

``start_run`` is the only write path: one transaction inserts the ``jobs`` row (through
``start_job``, whose ``with_rows`` callback inserts the ``training_runs`` row in the same
transaction), then the message goes to ``gpu_train``. Reads join the run with its job
(:mod:`src.db.repos.training_runs`); the log is a blob under the run's ``artifact_prefix``.

The run id on the wire is the job id. Lineage (export identity, registry pin, build sha) is read
once here and stored in ``training_runs.spec`` so the trainer and promote see what the run was
started with.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

from src.clients.storage.roots import build_rooted_store
from src.config import get_curation_config
from src.config.project_context import current_project, current_project_id
from src.config.storage import storage_root_paths
from src.core.logging import get_logger
from src.db.repos import training_runs as runs_repo
from src.db.session import async_session_factory, project_tx
from src.domain.training_keys import (
    REGISTRY_SNAPSHOT_NAME,
    RUN_LOG_NAME,
    run_object_key,
    run_prefix,
)
from src.services.jobs import service as job_service
from src.services.training import lineage
from src.services.training.run_wire import to_status


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repos.jobs import JobRow
    from src.services.training.job_models import TrainJobSpec, TrainJobStatus


logger = get_logger(__name__)

TRAIN_KIND = 'train'


def parse_run_id(run_id: str) -> uuid.UUID:
    """The job id a run is addressed by; anything else is a bad request."""
    try:
        return uuid.UUID(run_id)
    except ValueError:
        msg = f'invalid job_id: {run_id!r}'
        raise ValueError(msg) from None


def run_store() -> Any:
    """The blob store the run log and artifacts live in."""
    return build_rooted_store(storage_root_paths())


def _class_names(registry: Any, spec: TrainJobSpec) -> list[str]:
    """The model's class names in output order: the trainer renumbers a subset densely in id
    order, collapses ``single_cls`` to one ``object`` class, and trains a whole export in
    registry id order."""
    if spec.single_cls:
        return ['object']
    live = {c.class_id: c.class_name for c in registry.classes if not c.deprecated}
    ids = sorted(spec.include_classes) if spec.include_classes else sorted(live)
    missing = [i for i in ids if i not in live]
    if missing:
        msg = f'include_classes not in the class registry: {missing}'
        raise ValueError(msg)
    return [live[i] for i in ids]


async def _prepare(spec: TrainJobSpec) -> tuple[TrainJobSpec, Any, str]:
    """Fill the lineage fields; returns the spec, the pinned registry and its content."""
    from src.services.curation.registry_access import current_registry

    out = spec.model_copy(deep=True)
    if not out.dataset_export_dir:
        msg = 'dataset_export_dir is required (no current export exists to default to)'
        raise ValueError(msg)
    cfg = get_curation_config()
    out.project = cfg.project_slug
    out.project_export_root = str(cfg.export_root)
    out.mlflow_experiment = cfg.mlflow_experiment
    if out.dataset_sha is None or out.frozen_test_sha is None or out.test_label_sha is None:
        identity = await asyncio.to_thread(lineage.read_export_identity, out.dataset_export_dir)
        out.dataset_sha = out.dataset_sha or identity.dataset_sha
        out.frozen_test_sha = out.frozen_test_sha or identity.frozen_test_sha
        out.test_label_sha = out.test_label_sha or identity.test_label_sha
        out.dataset_version_tag = out.dataset_version_tag or identity.version_tag
    if out.api_sha is None:
        out.api_sha = lineage.api_sha()
    registry = await current_registry()
    content = registry.model_dump_json(indent=2)
    out.registry_sha = hashlib.sha256(content.encode('utf-8')).hexdigest()
    return out, registry, content


async def start_run(
    session: AsyncSession,
    spec: TrainJobSpec,
    *,
    campaign_id: str | None = None,
    requested_by: str = 'api',
) -> str:
    """Queue one training run and answer its id (the job id)."""
    prepared, registry, registry_json = await _prepare(spec)
    class_names = _class_names(registry, prepared)
    project = current_project_id()
    slug = current_project().record.slug
    params = {
        'campaign_id': campaign_id,
        'model_family': prepared.model_family,
        'model_size': prepared.model_size,
        'profile': prepared.profile,
    }

    async def with_rows(tx: AsyncSession, job: JobRow) -> None:
        prefix = run_prefix(slug, str(job.id))
        snapshot_key = run_object_key(prefix, REGISTRY_SNAPSHOT_NAME)
        store = run_store()
        await asyncio.to_thread(store.write_atomic, snapshot_key, registry_json.encode('utf-8'))
        prepared.registry_snapshot_path = str(store.local_path(snapshot_key))
        prepared.campaign_id = campaign_id
        await tx.run_sync(
            lambda sync: runs_repo.insert_run(
                sync,
                project_id=project,
                job_id=job.id,
                spec=prepared.model_dump(mode='json', exclude_none=False),
                class_names=class_names,
                family=prepared.model_family,
                base_model=str(
                    prepared.hyperparameters.get('model')
                    or f'{prepared.model_family}{prepared.model_size}'
                ),
                artifact_prefix=prefix,
                gpu_uuids=[],
            )
        )

    started = await job_service.start_job(
        session,
        TRAIN_KIND,
        project,
        params=params,
        requested_by=requested_by,
        with_rows=with_rows,
    )
    logger.info(
        'curation_train_run_queued',
        job_id=str(started.job.id),
        campaign_id=campaign_id,
        family=prepared.model_family,
        size=prepared.model_size,
        profile=prepared.profile,
    )
    return str(started.job.id)


async def cancel_run(job_id: str) -> bool:
    """Cancel one run (at once while queued, at the next epoch boundary while running).

    ``False`` when the project has no such run; a finished run answers 409 ``job_finished``.
    """
    run_id = parse_run_id(job_id)
    project = current_project_id()
    async with async_session_factory()() as session:
        if await _get_run(session, run_id) is None:
            return False
        await job_service.cancel_job(session, run_id, project, requested_by='api')
    return True


async def _get_run(session: AsyncSession, run_id: uuid.UUID) -> runs_repo.RunRow | None:
    async with project_tx(session, current_project_id()):
        return await session.run_sync(lambda sync: runs_repo.get(sync, run_id))


async def read_status(job_id: str) -> TrainJobStatus | None:
    """The run's status as the wire serves it, or ``None`` when the project has no such run."""
    run_id = parse_run_id(job_id)
    async with async_session_factory()() as session:
        run = await _get_run(session, run_id)
    return None if run is None else to_status(run)


async def list_runs(limit: int = 50, offset: int = 0) -> list[TrainJobStatus]:
    """Up to ``limit`` runs, newest first."""
    if limit <= 0:
        return []
    async with (
        async_session_factory()() as session,
        project_tx(session, current_project_id()),
    ):
        rows = await session.run_sync(
            lambda sync: runs_repo.list_page(sync, limit=limit, offset=offset)
        )
    return [to_status(r) for r in rows]


async def count_runs() -> int:
    async with (
        async_session_factory()() as session,
        project_tx(session, current_project_id()),
    ):
        return await session.run_sync(runs_repo.count)


async def get_active_job() -> TrainJobStatus | None:
    """The running run, else the oldest queued one, else the most recent finished run."""
    async with (
        async_session_factory()() as session,
        project_tx(session, current_project_id()),
    ):
        active = await session.run_sync(runs_repo.active)
        rows = active or await session.run_sync(
            lambda sync: runs_repo.list_page(sync, limit=1, offset=0)
        )
    return to_status(rows[0]) if rows else None


async def tail_run_log(job_id: str, lines: int = 200) -> list[str]:
    """The last ``lines`` lines of the run log, read from the end so a large log is not loaded."""
    run_id = parse_run_id(job_id)
    if lines <= 0:
        return []
    async with async_session_factory()() as session:
        run = await _get_run(session, run_id)
    if run is None:
        return []
    key = run_object_key(run.artifact_prefix, RUN_LOG_NAME)
    store = run_store()

    def _do() -> list[str]:
        if not store.exists(key):
            return []
        try:
            with store.open_read(key) as fh:
                fh.seek(0, 2)
                size = fh.tell()
                blocks: deque[bytes] = deque()
                read = 0
                while read < size and b''.join(blocks).count(b'\n') <= lines:
                    chunk = min(8192, size - read)
                    fh.seek(size - read - chunk)
                    blocks.appendleft(fh.read(chunk))
                    read += chunk
        except OSError as exc:
            logger.warning('curation_train_log_read_failed', job_id=job_id, error=str(exc))
            return []
        return b''.join(blocks).decode('utf-8', errors='replace').splitlines()[-lines:]

    return await asyncio.to_thread(_do)
