#!/usr/bin/env python3
"""Dedicated auto_label / recluster worker — own container, own logs.

Replaces the per-request ``subprocess.Popen`` model in
:mod:`src.services.curation.autolabel.job` with a long-lived worker
process. The yolo-api just drops a JSON trigger file under
``/jobs/auto_label/trigger.json`` and returns 202; this worker picks
it up, runs the pipeline, and writes state.json updates the SSE
endpoint already streams to the dashboard.

Why a dedicated container (vs. asyncio task / per-request subprocess):

* **Logs**: every Python exception, every stage transition, every
  OpenSearch error lands in ``docker compose logs
  curation-auto-label-worker`` — no more 'subprocess vanished before
  writing terminal state' opacity.
* **Survives API restarts**: stop/rm/up yolo-api does not kill an
  in-flight recluster; the worker keeps running.
* **Lifecycle ownership**: docker compose owns restart-on-crash via
  ``restart: unless-stopped`` — the API no longer races to detect a
  dead subprocess via ``/proc/<pid>/stat`` (which doesn't work
  across container boundaries anyway).
* **No PID-namespace gymnastics**: heartbeat file + mtime is the
  liveness signal both the worker (writes) and the API (reads) can
  trust.

Lifecycle contract with :mod:`auto_label_job`:

* Trigger file ``/jobs/auto_label/trigger.json`` — JSON payload
  ``{job_id, pipeline, args}`` written by ``start_job``. Worker
  claims it via ``unlink`` (atomic on Linux) then runs the pipeline.
* State file ``/jobs/auto_label/state.json`` — worker writes this as
  it advances stages; SSE reader (yolo-api) tails it via inotify.
* Heartbeat file ``/jobs/auto_label/heartbeat`` — worker touches
  every 5 s while running; API uses mtime to detect a dead worker.
* Cancel flag ``/jobs/auto_label/cancel.flag`` — operator writes via
  POST /curation/pipeline/auto_label/cancel; checked at stage boundaries.

Usage:

    python -m scripts.curation.auto_label_worker

Or via compose: ``docker compose up -d curation-auto-label-worker``.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import logging
import os
import signal
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any


# Make the repo root importable when run as a module from /app/ in the
# container (the deployment compose mounts ./scripts and ./src into /app/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.services.curation.autolabel.job import (
    _CANCEL_FLAG,
    _RUNNING_LOCK,
    _STATE_DIR,
    _STATE_FILE,
    _atomic_write,
    _ensure_dir,
    _JobState,
    _Progress,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-7s %(name)s | %(message)s',
)
logger = logging.getLogger('auto_label_worker')


TRIGGER_FILE = _STATE_DIR / 'trigger.json'
HEARTBEAT_FILE = _STATE_DIR / 'heartbeat'

# How often the worker checks for new trigger files when idle. Triggers
# are rare (operator-initiated); 1 s is responsive enough and uses no
# meaningful CPU. inotify would shave ~500 ms of average latency but
# add a dep we don't need.
POLL_INTERVAL_S = 1.0

# Heartbeat cadence — worker touches HEARTBEAT_FILE this often while a
# run is active. The API's liveness check uses 3x this as the stale
# threshold (worker considered dead if heartbeat hasn't been touched in
# 15 s). Conservative so a brief blocking call inside the pipeline
# doesn't false-positive a "worker is dead" verdict.
HEARTBEAT_INTERVAL_S = 5.0

# How often (seconds) the idle worker checks whether the IVF centroids
# should be retrained. The *decision* (should_retrain_centroids) has its
# own 24h cooldown + growth gate, so this is just how often we evaluate
# that cheap count query — 30 min keeps it responsive to a big ingest
# without spamming OpenSearch. 0 disables auto-retrain entirely.
AUTO_RETRAIN_CHECK_INTERVAL_S = float(os.getenv('OP_IVF_RETRAIN_CHECK_S', '1800'))

_IVF_PIPELINE_PATH = 'src.routers.curation.pipeline:pipeline_auto_label'


def _resolve_pipeline_fn(pipeline_path: str):
    """``module:qualname`` -> callable."""
    mod_name, _, attr_path = pipeline_path.partition(':')
    if not mod_name or not attr_path:
        raise ValueError(f'malformed pipeline path: {pipeline_path!r}')
    mod = importlib.import_module(mod_name)
    obj: Any = mod
    for part in attr_path.split('.'):
        obj = getattr(obj, part)
    return obj


async def _build_opensearch():
    """Async OpenSearch client.

    The curation pipeline code talks raw search/bulk/indices — pass the
    inner ``AsyncOpenSearch`` instance, not the project's higher-level
    ``OpenSearchClient`` wrapper. Mirrors the same pattern
    ``auto_label_cli._build_opensearch`` used.
    """
    from src.core.dependencies import OpenSearchClientFactory

    wrapper = await OpenSearchClientFactory.get_client()
    return getattr(wrapper, 'client', wrapper)


def _touch_heartbeat() -> None:
    """Update HEARTBEAT_FILE mtime — used by the API to detect a dead worker."""
    try:
        HEARTBEAT_FILE.touch()
    except OSError as exc:
        logger.warning('heartbeat touch failed: %s', exc)


async def _heartbeat_loop(stop: asyncio.Event) -> None:
    """Periodically touch HEARTBEAT_FILE until ``stop`` is set.

    Runs as a sibling task to the pipeline coroutine so a long-running
    blocking call inside a stage doesn't stall the heartbeat (the API's
    liveness check would otherwise false-positive 'worker dead').
    """
    while not stop.is_set():
        _touch_heartbeat()
        try:
            await asyncio.wait_for(stop.wait(), timeout=HEARTBEAT_INTERVAL_S)
        except TimeoutError:
            continue


async def _run_one(trigger: dict[str, Any], opensearch: Any) -> None:
    """Drive one pipeline invocation from a trigger payload."""
    job_id = trigger.get('job_id') or uuid.uuid4().hex
    pipeline_path = trigger.get('pipeline')
    args = dict(trigger.get('args') or {})
    # `opensearch` / `progress` are injected by us; never honor a caller
    # value (those serialize to garbage anyway).
    args.pop('opensearch', None)
    args.pop('progress', None)

    if not pipeline_path:
        logger.error('trigger missing pipeline path: %r', trigger)
        return

    state = _JobState(
        job_id=job_id,
        status='running',
        stage='',
        started_at=time.time(),
        args=args,
        pipeline=pipeline_path,
    )
    _atomic_write(asdict(state))
    # running.lock kept for backward-compat with any external tooling
    # that inspects it. cross-container pid is meaningless here, hence
    # the literal 'auto_label_worker' sentinel rather than os.getpid.
    with contextlib.suppress(OSError):
        _RUNNING_LOCK.write_text(
            json.dumps(
                {
                    'pid': os.getpid(),
                    'starttime': 0,
                    'started_at': state.started_at,
                    'owner': 'auto_label_worker',
                }
            )
        )

    try:
        pipeline_fn = _resolve_pipeline_fn(pipeline_path)
    except (ImportError, AttributeError, ValueError) as exc:
        state.status = 'failed'
        state.error = f'cannot resolve pipeline {pipeline_path!r}: {exc}'
        state.error_detail = traceback.format_exc()[:4096]
        state.finished_at = time.time()
        _atomic_write(asdict(state))
        logger.exception('pipeline resolve failed')
        return

    stop = asyncio.Event()
    heartbeat_task = asyncio.create_task(_heartbeat_loop(stop))
    progress = _Progress(state)

    logger.info(
        'run starting job_id=%s pipeline=%s args=%s',
        job_id,
        pipeline_path,
        args,
    )
    try:
        result = await pipeline_fn(opensearch=opensearch, progress=progress, **args)
        state.result = result if isinstance(result, dict) else {'raw': str(result)}
        state.status = 'completed'
        logger.info('run completed job_id=%s', job_id)
    except asyncio.CancelledError:
        state.status = 'cancelled'
        state.error = state.error or 'cancelled by operator'
        logger.info('run cancelled job_id=%s', job_id)
        raise
    except BaseException as exc:
        state.status = 'failed'
        state.error = f'{type(exc).__name__}: {exc}'
        state.error_detail = traceback.format_exc()[:4096]
        logger.exception('run failed job_id=%s', job_id)
    finally:
        stop.set()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
        # Flush the final stage's duration so the dashboard's
        # "stage timings" expander has a complete row.
        with contextlib.suppress(Exception):
            progress.finalize()
        state.finished_at = time.time()
        _atomic_write(asdict(state))
        with contextlib.suppress(FileNotFoundError):
            _RUNNING_LOCK.unlink()
        with contextlib.suppress(FileNotFoundError):
            _CANCEL_FLAG.unlink()
        with contextlib.suppress(FileNotFoundError):
            HEARTBEAT_FILE.unlink()


def _claim_trigger() -> dict[str, Any] | None:
    """Atomically claim the trigger file. Returns the parsed payload or None.

    Reads + unlinks in one critical section. Multiple workers running
    against the same volume would each see the file once at most;
    whichever wins the unlink race processes the trigger. In production
    we run only one worker, so this is just defense-in-depth.
    """
    if not TRIGGER_FILE.exists():
        return None
    try:
        raw = TRIGGER_FILE.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning('trigger read failed: %s', exc)
        return None
    with contextlib.suppress(FileNotFoundError):
        TRIGGER_FILE.unlink()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error('trigger JSON invalid: %s; raw=%r', exc, raw[:200])
        return None


def _ivf_retrain_trigger() -> dict[str, Any]:
    """Synthesize a self-trigger for a full IVF retrain + reassign."""
    return {
        'job_id': uuid.uuid4().hex,
        'pipeline': _IVF_PIPELINE_PATH,
        'args': {
            'train_clusters': True,
            'clustering_method': 'ivf',
            'recluster_unvalidated': True,
            'run_gemma': False,
            'run_auto_promote': False,
        },
    }


async def _maybe_auto_retrain(opensearch: Any) -> dict[str, Any] | None:
    """Return a retrain trigger if IVF centroids are stale, else None.

    Cheap count query; the growth + 24h cooldown gate lives in
    should_retrain_centroids. Failures are swallowed (logged) so a
    transient OpenSearch hiccup never crashes the idle loop.
    """
    try:
        from src.services.curation.clustering.orchestrator import should_retrain_centroids

        decision = await should_retrain_centroids(opensearch)
        if decision.get('should'):
            logger.info('ivf auto-retrain triggered: %s', decision)
            return _ivf_retrain_trigger()
        logger.debug('ivf auto-retrain skipped: %s', decision)
    except Exception as exc:
        logger.warning('ivf auto-retrain check failed: %s', exc)
    return None


async def _main_loop(stop: asyncio.Event) -> None:
    """Long-lived loop: open OpenSearch client, watch trigger file, dispatch."""
    _ensure_dir()
    opensearch = await _build_opensearch()
    logger.info(
        'worker ready trigger=%s state=%s heartbeat=%s poll_s=%s retrain_check_s=%s',
        TRIGGER_FILE,
        _STATE_FILE,
        HEARTBEAT_FILE,
        POLL_INTERVAL_S,
        AUTO_RETRAIN_CHECK_INTERVAL_S,
    )

    # Stagger the first auto-retrain check so a worker restart during a
    # big ingest doesn't immediately fire one. Next check after one
    # interval.
    last_retrain_check = time.monotonic()

    try:
        while not stop.is_set():
            trigger = _claim_trigger()
            # Idle-time auto-retrain: only when no operator trigger is
            # pending, throttled by AUTO_RETRAIN_CHECK_INTERVAL_S (the
            # decision itself enforces the growth + cooldown gate).
            if (
                trigger is None
                and AUTO_RETRAIN_CHECK_INTERVAL_S > 0
                and time.monotonic() - last_retrain_check >= AUTO_RETRAIN_CHECK_INTERVAL_S
            ):
                last_retrain_check = time.monotonic()
                trigger = await _maybe_auto_retrain(opensearch)
            if trigger is not None:
                try:
                    await _run_one(trigger, opensearch)
                except asyncio.CancelledError:
                    # Worker shutdown — the in-flight run already
                    # recorded 'cancelled' state in its own finally.
                    raise
                continue
            try:
                await asyncio.wait_for(stop.wait(), timeout=POLL_INTERVAL_S)
            except TimeoutError:
                continue
    finally:
        with contextlib.suppress(Exception):
            await opensearch.close()


def main() -> int:
    stop = asyncio.Event()

    def _on_signal(*_: object) -> None:
        if not stop.is_set():
            logger.info('shutdown signal received')
            stop.set()

    async def _amain() -> int:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(sig, _on_signal)
        with contextlib.suppress(asyncio.CancelledError):
            await _main_loop(stop)
        return 0

    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
