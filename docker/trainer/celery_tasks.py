"""The trainer's Celery entry: the ``train`` job handler on the ``gpu_train`` queue.

Run as ``celery -A celery_tasks:app worker -Q gpu_train -c 1 --max-tasks-per-child=1``: every run
gets a fresh pool child, so GPU memory is released when it ends. The job runtime
(``src/tasks/runtime``) claims the row, heartbeats and writes the terminal state; this module
turns the run's ``training_runs`` row into a :class:`~job_spec.JobSpec`, runs the blocking fit in
a worker thread and, meanwhile, publishes progress, the run log and the cancel check from the
event loop. Cancel is cooperative: ``ctx.cancelled()`` is polled here and set on the run's
state, which the epoch callback reads at the next epoch boundary.

The image imports only ``src.tasks.runtime``, ``src.db``, ``src.config``, ``src.domain``,
``src.utils`` and the storage client; never FastAPI or the OpenSearch client
(``tests/trainer/test_runtime_import_minimal.py``).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from campaign import maybe_handle_campaign, write_quant_bakeoff_job
from celery.signals import worker_init
from dataset_prep import read_class_remap
from job_manifest import build_manifest
from job_spec import JobSpec, parse_and_validate_spec
from job_status import StatusState, progress_counters
from logutil import get_logger
from run_log import RunLog, tee_to_run_log
from sqlalchemy.exc import SQLAlchemyError
from trainer import run_training

from src.clients.storage.roots import build_rooted_store
from src.config.storage import storage_root_paths
from src.db.repos import gpu_devices, gpu_reservations as reservations, training_runs as runs_repo
from src.domain.gpu_order import parse_order, position_of
from src.domain.training_keys import RUN_ARTIFACT_WHITELIST, RUN_LOG_NAME, run_object_key
from src.tasks.runtime.gpu_app import make_app
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.clients.storage.base import BlobStore
    from src.tasks.runtime.context import JobContext


logger = get_logger('trainer.celery_tasks')

POLL_INTERVAL_S = 2.0

app = make_app(include=['celery_tasks'])


@worker_init.connect(weak=False)
def _on_worker_init(**_: Any) -> None:
    """Refuse to serve on a retired setting, an unreachable database or another schema revision.

    Celery logs an ``Exception`` raised by a signal receiver and carries on, so the refusal is
    a ``SystemExit`` (https://docs.celeryq.dev/en/stable/userguide/signals.html).
    """
    from src.config.retired_env import reject_retired_env
    from src.db import schema_check

    reject_retired_env()
    schema_check.require_current_schema()


async def _load_run(ctx: JobContext) -> runs_repo.RunRow:
    async with ctx.session() as session:
        run = await session.run_sync(lambda sync: runs_repo.get(sync, ctx.job_id))
    if run is None:
        msg = f'job {ctx.job_id} has no training_runs row'
        raise LookupError(msg)
    return run


async def _publish_until_done(
    ctx: JobContext, fit: asyncio.Future[None], state: StatusState, log: RunLog
) -> None:
    """Every ``POLL_INTERVAL_S`` while the fit thread runs: relay a cancel request to the epoch
    callback and publish the epoch progress and the log."""
    while True:
        done, _ = await asyncio.wait({fit}, timeout=POLL_INTERVAL_S)
        if ctx.cancelled():
            state.cancel_requested.set()
        counters = await asyncio.to_thread(progress_counters, state)
        with state.lock:
            epoch, total, stage = state.current_epoch, state.total_epochs, state.state
        await ctx.progress(epoch, total or None, stage, counters)
        await asyncio.to_thread(log.publish)
        if done:
            return


def _run_dir(state: StatusState) -> Path | None:
    """Ultralytics' run directory, from the checkpoint (``<run>/weights/best.pt``) or, for a run
    that failed before export, from the confusion matrix."""
    if state.checkpoint_path:
        return Path(state.checkpoint_path).parent.parent
    cm = (state.eval or {}).get('confusion_matrix_path')
    return Path(cm).parent if isinstance(cm, str) and cm else None


def _store_artifacts(state: StatusState, store: BlobStore, prefix: str) -> None:
    """Copy the whitelisted metric files of the run directory under the run's artifact prefix."""
    run_dir = _run_dir(state)
    if run_dir is None:
        return
    for name in sorted(RUN_ARTIFACT_WHITELIST):
        source = run_dir / name
        if source.is_file():
            store.write_atomic(run_object_key(prefix, name), source.read_bytes())


async def _record_end(
    ctx: JobContext, spec: JobSpec, state: StatusState, run: runs_repo.RunRow, store: BlobStore
) -> None:
    """Store the manifest and metric artifacts, then run the campaign hooks (best effort)."""
    manifest = build_manifest(spec, state, class_remap=read_class_remap(spec))
    await asyncio.to_thread(_store_artifacts, state, store, run.artifact_prefix)
    if manifest is not None:
        async with ctx.session() as session:
            await session.run_sync(
                lambda sync: runs_repo.set_manifest(sync, ctx.job_id, manifest, state.mlflow_run_id)
            )
    if spec.auto_quantize_bakeoff and state.state == 'finished':
        await asyncio.to_thread(write_quant_bakeoff_job, spec, state)  # logs its own failures
    try:
        await maybe_handle_campaign(ctx, spec, state)
    except (SQLAlchemyError, OSError) as exc:  # sibling bookkeeping is non-fatal
        logger.warning('campaign hook failed', job_id=spec.job_id, error=str(exc))


async def _reserve_cards(ctx: JobContext, spec: JobSpec) -> list[str]:
    """Hold the cards the run names, by UUID, and record them on the run.

    The run names host card numbers; ``OP_TRAIN_GPU_ORDER`` maps each to its position in this
    container, and ``gpu_devices`` (written by this worker at start) maps the position to the
    card's UUID, so a CUDA/NVML numbering that differs from the host's never names another card.
    """
    host_ids = [int(t) for t in (spec.cuda_visible_devices or '').split(',') if t.strip()]
    if not host_ids:
        return []
    order = parse_order(os.environ.get('OP_TRAIN_GPU_ORDER'))
    positions = [position_of(order, host_id) for host_id in host_ids]
    host = socket.gethostname()
    async with ctx.session() as session:
        cards = await session.run_sync(lambda sync: gpu_devices.on_host_at(sync, host, positions))
        if len(cards) != len(positions):
            msg = f'cards at positions {positions} are not recorded for host {host}'
            raise LookupError(msg)
        await session.run_sync(lambda sync: reservations.reserve(sync, ctx.job_id, cards))
        uuids = [card.gpu_uuid for card in cards]
        await session.run_sync(lambda sync: runs_repo.set_gpu_uuids(sync, ctx.job_id, uuids))
    return uuids


async def _release_cards(ctx: JobContext) -> None:
    async with ctx.session() as session:
        await session.run_sync(lambda sync: reservations.release(sync, ctx.job_id))


async def train(ctx: JobContext) -> dict[str, Any]:
    """Run one training run; an exception ends the job ``failed`` with its message."""
    run = await _load_run(ctx)
    raw = {**run.spec, 'job_id': str(ctx.job_id)}
    raw.setdefault('mlflow_run_name', str(ctx.job_id))
    spec = parse_and_validate_spec(str(ctx.job_id), raw)
    store = build_rooted_store(storage_root_paths())
    state = StatusState(job_id=spec.job_id, campaign_id=spec.campaign_id, state='starting')
    log = RunLog(spec.run_log_path, store, run_object_key(run.artifact_prefix, RUN_LOG_NAME))

    try:
        with tee_to_run_log(log.scratch):
            await _reserve_cards(ctx, spec)
            fit = asyncio.ensure_future(asyncio.to_thread(run_training, spec, state))
            await _publish_until_done(ctx, fit, state, log)
            fit.result()
    finally:
        # The exception in flight, if any, is recorded in the run log and the manifest and then
        # re-raised by the ``finally`` itself, so the job ends ``failed`` with its message.
        failure = sys.exc_info()[1]
        if failure is not None:
            with state.lock:
                state.state = 'failed'
                state.error = f'{type(failure).__name__}: {failure}'
            log.append(f'\n[trainer] FAILED: {failure}\n{traceback.format_exc()}\n')
        try:
            await _record_end(ctx, spec, state, run, store)
            await asyncio.to_thread(log.publish)
        finally:
            await _release_cards(ctx)
            shutil.rmtree(spec.tmp_root, ignore_errors=True)
    return {'state': state.state, 'checkpoint_path': state.checkpoint_path}


register_job('train', train)
