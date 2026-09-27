#!/usr/bin/env python3
"""Dedicated auto_label / recluster worker — own container, own logs.

Replaces the per-request ``subprocess.Popen`` model in
:mod:`src.services.curation.autolabel.job` with a long-lived worker
process. The yolo-api just drops a JSON trigger file under
``<project's autolabel_dir>/trigger.json`` and returns 202; this
worker picks it up, runs the pipeline, and writes state.json updates
the SSE endpoint already streams to the dashboard.

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

Lifecycle contract with :mod:`auto_label_job`, all paths under the
owning project's own ``autolabel_dir``:

* Trigger file ``trigger.json`` — JSON payload
  ``{job_id, pipeline, args}`` written by ``start_job``. Worker
  claims it via ``unlink`` (atomic on Linux) then runs the pipeline.
* State file ``state.json`` — worker writes this as it advances
  stages; SSE reader (yolo-api) tails it via inotify.
* Heartbeat file ``heartbeat`` — worker touches every 5 s while
  running; API uses mtime to detect a dead worker.
* Cancel flag ``cancel.flag`` — operator writes via
  POST /curation/projects/{slug}/pipeline/auto_label/cancel; checked
  at stage boundaries.

Multi-project mode (default, no ``--project``): every poll cycle,
scans every ``active`` project's own ``autolabel_dir`` for a pending
``trigger.json``, and runs only the single OLDEST one (by trigger
mtime) this cycle — never two projects' jobs concurrently (the
pipeline is GPU/CPU heavy). The other pending triggers wait for the
next cycle. ``--project SLUG`` restricts discovery to one project
(and whole-process-binds it) -- a dedicated per-project worker
container.

Usage:

    python -m scripts.curation.auto_label_worker
    python -m scripts.curation.auto_label_worker --project my-project

Or via compose: ``docker compose up -d curation-auto-label-worker``.
"""

from __future__ import annotations

import argparse
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
from typing import TYPE_CHECKING, Any


# Make the repo root importable when run as a module from /app/ in the
# container (the deployment compose mounts ./scripts and ./src into /app/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.services.curation.autolabel.job import (
    _atomic_write,
    _cancel_flag,
    _ensure_dir,
    _heartbeat_file,
    _JobState,
    _Progress,
    _running_lock,
    _trigger_file,
)
from src.services.curation.worker_liveness import write_heartbeat as _write_container_heartbeat
from src.services.projects.script_binding import add_project_argument, bind_script_project


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-7s %(name)s | %(message)s',
)
logger = logging.getLogger('auto_label_worker')


# How often the worker checks for new trigger files when idle. Triggers
# are rare (operator-initiated); 1 s is responsive enough and uses no
# meaningful CPU. inotify would shave ~500 ms of average latency but
# add a dep we don't need.
POLL_INTERVAL_S = 1.0

# Heartbeat cadence — worker touches the active project's heartbeat file
# this often while a run is active. The API's liveness check uses 3x
# this as the stale threshold (worker considered dead if heartbeat
# hasn't been touched in 15 s). Conservative so a brief blocking call
# inside the pipeline doesn't false-positive a "worker is dead" verdict.
HEARTBEAT_INTERVAL_S = 5.0

# How often (seconds) the idle worker checks whether the IVF centroids
# should be retrained (single-project mode only -- see main()'s
# docstring). 0 disables auto-retrain entirely.
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
    """Update the bound project's heartbeat-file mtime -- used by the API
    to detect a dead worker. Must be called with a project bound.

    Also writes the container-local liveness heartbeat (S-2) so the
    compose healthcheck stays fresh for the whole duration of a run, not
    just the idle poll loop.
    """
    try:
        _heartbeat_file().touch()
    except OSError as exc:
        logger.warning('heartbeat touch failed: %s', exc)
    _write_container_heartbeat('auto_label_worker', {'poll': True})


async def _heartbeat_loop(stop: asyncio.Event) -> None:
    """Periodically touch the heartbeat file until ``stop`` is set.

    Runs as a sibling task to the pipeline coroutine so a long-running
    blocking call inside a stage doesn't stall the heartbeat. Created
    while a project is bound, so it inherits that binding (asyncio
    tasks capture a copy of the current context at creation time)."""
    while not stop.is_set():
        _touch_heartbeat()
        try:
            await asyncio.wait_for(stop.wait(), timeout=HEARTBEAT_INTERVAL_S)
        except TimeoutError:
            continue


async def _run_one(record: ProjectRecord, trigger: dict[str, Any], opensearch: Any) -> None:
    """Drive one pipeline invocation from a trigger payload, with
    ``record`` bound for the whole run (including the sibling heartbeat
    task, state/cancel/lock file resolution, and the pipeline itself)."""
    from src.config.project_context import bind_project

    job_id = trigger.get('job_id') or uuid.uuid4().hex
    pipeline_path = trigger.get('pipeline')
    args = dict(trigger.get('args') or {})
    # `opensearch` / `progress` are injected by us; never honor a caller
    # value (those serialize to garbage anyway).
    args.pop('opensearch', None)
    args.pop('progress', None)

    if not pipeline_path:
        logger.error('project=%s trigger missing pipeline path: %r', record.slug, trigger)
        return

    with bind_project(record):
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
            _running_lock().write_text(
                json.dumps(
                    {
                        'pid': os.getpid(),
                        'starttime': 0,
                        'started_at': state.started_at,
                        'owner': 'auto_label_worker',
                        'project': record.slug,
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
            logger.exception('project=%s pipeline resolve failed', record.slug)
            return

        stop = asyncio.Event()
        heartbeat_task = asyncio.create_task(_heartbeat_loop(stop))
        progress = _Progress(state)

        logger.info(
            'project=%s run starting job_id=%s pipeline=%s args=%s',
            record.slug,
            job_id,
            pipeline_path,
            args,
        )
        try:
            result = await pipeline_fn(opensearch=opensearch, progress=progress, **args)
            state.result = result if isinstance(result, dict) else {'raw': str(result)}
            state.status = 'completed'
            logger.info('project=%s run completed job_id=%s', record.slug, job_id)
        except asyncio.CancelledError:
            state.status = 'cancelled'
            state.error = state.error or 'cancelled by operator'
            logger.info('project=%s run cancelled job_id=%s', record.slug, job_id)
            raise
        except BaseException as exc:
            state.status = 'failed'
            state.error = f'{type(exc).__name__}: {exc}'
            state.error_detail = traceback.format_exc()[:4096]
            logger.exception('project=%s run failed job_id=%s', record.slug, job_id)
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
                _running_lock().unlink()
            with contextlib.suppress(FileNotFoundError):
                _cancel_flag().unlink()
            with contextlib.suppress(FileNotFoundError):
                _heartbeat_file().unlink()


def _claim_trigger_for(record: ProjectRecord) -> dict[str, Any] | None:
    """Atomically claim ``record``'s trigger file (read + unlink in one
    critical section, project bound). Returns the parsed payload or
    None if there's nothing pending (or it's invalid)."""
    from src.config.project_context import bind_project

    with bind_project(record):
        trigger_path = _trigger_file()
        if not trigger_path.exists():
            return None
        try:
            raw = trigger_path.read_text()
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning('project=%s trigger read failed: %s', record.slug, exc)
            return None
        with contextlib.suppress(FileNotFoundError):
            trigger_path.unlink()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error('project=%s trigger JSON invalid: %s; raw=%r', record.slug, exc, raw[:200])
        return None


def _oldest_pending_trigger(
    active: list[ProjectRecord],
) -> tuple[ProjectRecord, float] | None:
    """Among every active project with a pending (unclaimed)
    ``trigger.json``, the one whose trigger is oldest by mtime. Doesn't
    claim it -- callers still race a project's own worker in
    single-project mode, so claiming happens separately via
    :func:`_claim_trigger_for`."""
    from src.config.project_context import bind_project

    best: tuple[ProjectRecord, float] | None = None
    for record in active:
        try:
            with bind_project(record):
                trigger_path = _trigger_file()
                mtime = trigger_path.stat().st_mtime
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning('project=%s trigger discovery failed: %s', record.slug, exc)
            continue
        if best is None or mtime < best[1]:
            best = (record, mtime)
    return best


def _ivf_retrain_trigger() -> dict[str, Any]:
    """Synthesize a self-trigger for a full IVF retrain + reassign."""
    return {
        'job_id': uuid.uuid4().hex,
        'pipeline': _IVF_PIPELINE_PATH,
        'args': {
            'train_clusters': True,
            'clustering_method': 'ivf',
            'recluster_unvalidated': True,
            'run_vlm': False,
            'run_auto_promote': False,
        },
    }


async def _maybe_auto_retrain(opensearch: Any) -> dict[str, Any] | None:
    """Return a retrain trigger if IVF centroids are stale, else None.
    Single-project mode only -- see the module docstring.

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


async def _main_loop_single(stop: asyncio.Event, record: ProjectRecord) -> None:
    """Single, ``--project``-bound project: the original loop, plus IVF
    auto-retrain (multi-project mode does not do auto-retrain -- see
    module docstring)."""
    from src.config.project_context import bind_project

    with bind_project(record):
        _ensure_dir()
    opensearch = await _build_opensearch()
    logger.info(
        'worker ready project=%s poll_s=%s retrain_check_s=%s',
        record.slug,
        POLL_INTERVAL_S,
        AUTO_RETRAIN_CHECK_INTERVAL_S,
    )

    last_retrain_check = time.monotonic()
    last_liveness_heartbeat = 0.0
    try:
        while not stop.is_set():
            now_monotonic = time.monotonic()
            if now_monotonic - last_liveness_heartbeat >= 15.0:
                last_liveness_heartbeat = now_monotonic
                _write_container_heartbeat('auto_label_worker', {'poll': True})
            trigger = _claim_trigger_for(record)
            if (
                trigger is None
                and AUTO_RETRAIN_CHECK_INTERVAL_S > 0
                and time.monotonic() - last_retrain_check >= AUTO_RETRAIN_CHECK_INTERVAL_S
            ):
                last_retrain_check = time.monotonic()
                trigger = await _maybe_auto_retrain(opensearch)
            if trigger is not None:
                try:
                    await _run_one(record, trigger, opensearch)
                except asyncio.CancelledError:
                    raise
                continue
            try:
                await asyncio.wait_for(stop.wait(), timeout=POLL_INTERVAL_S)
            except TimeoutError:
                continue
    finally:
        with contextlib.suppress(Exception):
            await opensearch.close()


async def _main_loop_multi(stop: asyncio.Event) -> None:
    """Discovers every active project's pending trigger each cycle and
    runs only the single oldest one (by trigger mtime) -- never two
    projects' jobs concurrently. No IVF auto-retrain (see module
    docstring)."""
    from src.services.projects.registry import get_project_registry

    registry = get_project_registry()
    opensearch = await _build_opensearch()
    logger.info('worker ready (multi-project) poll_s=%s', POLL_INTERVAL_S)
    last_liveness_heartbeat = 0.0
    try:
        while not stop.is_set():
            now_monotonic = time.monotonic()
            if now_monotonic - last_liveness_heartbeat >= 15.0:
                last_liveness_heartbeat = now_monotonic
                _write_container_heartbeat('auto_label_worker', {'poll': True})

            try:
                await registry.ensure_fresh()
                active = registry.active_projects()
            except Exception as exc:
                logger.warning('registry unavailable: %s', exc)
                active = []

            oldest = _oldest_pending_trigger(active)
            if oldest is not None:
                record, _mtime = oldest
                trigger = _claim_trigger_for(record)
                if trigger is not None:
                    try:
                        await _run_one(record, trigger, opensearch)
                    except asyncio.CancelledError:
                        raise
                    continue
            try:
                await asyncio.wait_for(stop.wait(), timeout=POLL_INTERVAL_S)
            except TimeoutError:
                continue
    finally:
        with contextlib.suppress(Exception):
            await opensearch.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_project_argument(p)
    p.set_defaults(project=None)  # multi-project mode is the new default
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    record = None
    if args.project:
        record = bind_script_project(args.project)

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
            if record is not None:
                await _main_loop_single(stop, record)
            else:
                await _main_loop_multi(stop)
        return 0

    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
