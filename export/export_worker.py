"""Model-export worker: the export image's long-running entrypoint for API-queued builds.

The API image has no TensorRT, so ``POST /models/upload`` and ``POST /models/{name}/export``
queue a job file instead of exporting (``src/services/model_export.py`` documents the protocol).
This process claims each job, runs the export script as a subprocess, parses its output into
task progress, loads the engines into Triton when asked, and writes the task status the status
routes serve.

    python -m export.export_worker --watch /jobs/model_export          # the compose service
    python -m export.export_worker --watch /jobs/model_export --once   # drain the queue, exit
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess  # nosec B404 - runs the repo's own export scripts, never a user-supplied command
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from src.schemas.models import ExportStatus
from src.services import model_export as store
from src.services.triton_control import TritonControlService


logger = logging.getLogger('export_worker')

EXPORT_DIR = Path(__file__).resolve().parent
APP_DIR = EXPORT_DIR.parent
# Classic YOLO11-family toolchain (EfficientNMS; re-execs into /opt/venv-y11)
EXPORT_SCRIPT = EXPORT_DIR / 'export_models.py'
# Native NMS-free toolchain for one-to-one heads (YOLO26, YOLOv10)
EXPORT_SCRIPT_Y26 = EXPORT_DIR / 'export_yolo26.py'
POLL_INTERVAL_S = 2.0
HEARTBEAT_INTERVAL_S = 2.0
# A claimed task whose own heartbeat is older than this was left by a dead worker.
TASK_STALE_S = 30.0
# A TensorRT build of a large detector takes minutes; hours means the build is hung.
EXPORT_TIMEOUT_S = 4 * 3600.0
# A task file the API wrote this recently may still be waiting for its job file.
ORPHAN_GRACE_S = 300.0
_NON_TERMINAL = {
    ExportStatus.PENDING.value,
    ExportStatus.VALIDATING.value,
    ExportStatus.EXPORTING.value,
    ExportStatus.LOADING.value,
}


def write_heartbeat() -> None:
    beat = store.heartbeat_path()
    beat.parent.mkdir(parents=True, exist_ok=True)
    beat.write_text(str(time.time()))


# The task this worker is exporting now, so its heartbeat thread can mark that claim alive.
_running_task: str | None = None


def _task_heartbeat(task_id: str) -> Path:
    return store.claimed_dir() / f'{task_id}.heartbeat'


def _task_is_alive(task_id: str) -> bool:
    """A claimed job whose per-task heartbeat is fresh belongs to a running worker (maybe a
    sibling replica); a missing or old one belongs to a dead worker."""
    try:
        age = time.time() - _task_heartbeat(task_id).stat().st_mtime
    except OSError:
        return False
    return age <= TASK_STALE_S


def _heartbeat_loop(stop: threading.Event) -> None:
    while not stop.is_set():
        with contextlib.suppress(OSError):
            write_heartbeat()
            task_id = _running_task
            if task_id is not None:
                _task_heartbeat(task_id).touch()
        stop.wait(HEARTBEAT_INTERVAL_S)


def claim_next() -> dict[str, Any] | None:
    """Claim the oldest queued job by atomically renaming it into ``claimed/``."""
    queue = store.queue_dir()
    if not queue.is_dir():
        return None
    claimed = store.claimed_dir()
    claimed.mkdir(parents=True, exist_ok=True)
    for path in sorted(queue.glob('*.job.json'), key=lambda p: (p.stat().st_mtime, p.name)):
        target = claimed / path.name
        try:
            path.rename(target)
        except FileNotFoundError:
            continue  # another worker claimed it first
        try:
            job: dict[str, Any] = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.error('unreadable job file %s: %s', target, exc)
            _fail(path.name.removesuffix('.job.json'), f'unreadable job file: {exc}')
            target.unlink(missing_ok=True)
            continue
        return job
    return None


def reconcile_claimed() -> int:
    """Startup repair. A job still in ``claimed/`` belongs to a worker that died mid-export. A task
    left running with no job file at all (for example one an older, in-API export left behind
    at an upgrade) can never finish either, once it is older than the API's create-then-queue
    window."""
    claimed = store.claimed_dir()
    reconciled = 0
    if claimed.is_dir():
        for path in sorted(claimed.glob('*.job.json')):
            task_id = path.name.removesuffix('.job.json')
            if _task_is_alive(task_id):
                continue  # another worker is exporting it right now
            _fail(task_id, 'the model-export worker stopped while this export was running')
            path.unlink(missing_ok=True)
            _task_heartbeat(task_id).unlink(missing_ok=True)
            reconciled += 1
    store.load_tasks_from_disk()
    queued = store.queue_dir()
    now = datetime.now(UTC)
    for task_id, task in list(store.export_tasks.items()):
        status = getattr(task.get('status'), 'value', task.get('status'))
        if status not in _NON_TERMINAL or (queued / f'{task_id}.job.json').exists():
            continue
        created = task.get('created_at')
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        if isinstance(created, datetime) and (now - created).total_seconds() < ORPHAN_GRACE_S:
            continue
        _fail(task_id, 'no model-export job exists for this task; start the export again')
        reconciled += 1
    return reconciled


def _fail(task_id: str, error: str, started: float | None = None) -> None:
    task = store.get_export_task(task_id)
    if task is None:
        logger.error('no task file for job %s; nothing to mark failed', task_id)
        return
    task['status'] = ExportStatus.FAILED
    task['error'] = error
    task['message'] = f'Export failed: {error}'
    task['completed_at'] = datetime.now(UTC)
    if started is not None:
        task['export_duration_seconds'] = round(perf_counter() - started, 2)
    store.export_tasks[task_id] = task
    store.persist_task(task_id)


def _head_kind(job: dict[str, Any]) -> str:
    """The job's head kind. A job queued before head kinds existed carries none: fail closed."""
    kind = job.get('head_kind')
    if kind not in ('one2one', 'nms'):
        raise ValueError(
            'this export job has no head kind (it was queued by an older version): '
            're-upload the model'
        )
    return str(kind)


def build_command(job: dict[str, Any]) -> list[str]:
    """The export subprocess for a job, routed by head kind."""
    custom = f'{job["pt_file"]}:{job["triton_name"]}:{job["max_batch"]}'
    formats: list[str] = job['formats']
    if _head_kind(job) == 'one2one':
        # The NMS-free head (YOLO26, YOLOv10) is natively NMS-free: any TRT-flavored format request maps to its single fused
        # 'trt' output. Labels and config.pbtxt are always generated by this exporter.
        native = ['trt'] if any(f.startswith(('trt', 'all')) for f in formats) else ['onnx']
        return [
            sys.executable,
            str(EXPORT_SCRIPT_Y26),
            '--custom-model',
            custom,
            '--formats',
            *native,
        ]
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        '--custom-model',
        custom,
        '--formats',
        *formats,
        '--save-labels',
        '--generate-config',
    ]
    if job.get('normalize_boxes'):
        cmd.append('--normalize-boxes')
    return cmd


def engines_to_load(job: dict[str, Any]) -> list[str]:
    name, formats = job['triton_name'], job['formats']
    if _head_kind(job) == 'one2one':
        return [f'{name}_trt']
    models = []
    if 'trt' in formats or 'all' in formats:
        models.append(f'{name}_trt')
    if 'trt_end2end' in formats or 'all' in formats:
        models.append(f'{name}_trt_end2end')
    return models


def load_into_triton(names: list[str]) -> bool:
    """Load each engine into Triton; True when at least one loaded."""
    triton = TritonControlService()
    loaded = False
    for name in names:
        ok, msg = asyncio.run(triton.load_model(name))
        if ok:
            loaded = True
        else:
            logger.warning('could not load %s into Triton: %s', name, msg)
    return loaded


def _phase_of(line: str) -> tuple[str, float, str] | None:
    """(phase, progress, step text) for an export script output line that marks one."""
    if 'ONNX' in line and 'Export' in line:
        return 'onnx_export', 40.0, 'Exporting ONNX End2End'
    if 'TensorRT' in line or 'Building' in line:
        return 'tensorrt_build', 60.0, 'Building TensorRT engine (this takes 2-5 min)'
    if 'Engine saved' in line:
        return 'finalization', 85.0, 'Finalizing export'
    return None


def run_job(job: dict[str, Any]) -> None:
    """Run one claimed job to a terminal task status and release the claim."""
    global _running_task  # noqa: PLW0603 - the one task this worker runs, read by its heartbeat thread
    task_id = job['task_id']
    claim = store.claimed_dir() / f'{task_id}.job.json'
    started = perf_counter()
    _task_heartbeat(task_id).touch()
    _running_task = task_id
    try:
        task = store.get_export_task(task_id)
        if task is None:
            logger.error('job %s has no task file; dropping it', task_id)
            return
        _export(task_id, task, job, started)
    except Exception as exc:
        logger.exception('[%s] export failed', task_id)
        _fail(task_id, str(exc), started)
    finally:
        _running_task = None
        claim.unlink(missing_ok=True)
        _task_heartbeat(task_id).unlink(missing_ok=True)


def _export(task_id: str, task: dict[str, Any], job: dict[str, Any], started: float) -> None:
    step_times: dict[str, float] = {}
    task.update(
        status=ExportStatus.EXPORTING,
        started_at=datetime.now(UTC),
        current_step=f'Exporting: {", ".join(job["formats"])}',
        progress=20.0,
        step_times=step_times,
    )
    store.export_tasks[task_id] = task
    store.persist_task(task_id)

    cmd = build_command(job)
    logger.info('[%s] running: %s', task_id, ' '.join(cmd))
    phase, phase_start = 'initialization', perf_counter()
    output: list[str] = []
    with subprocess.Popen(  # nosec B603 - argv built from the repo's own scripts
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors='replace',
        cwd=APP_DIR,
        env={**os.environ, 'PYTHONPATH': str(APP_DIR)},
    ) as proc:
        assert proc.stdout is not None  # PIPE was requested above
        # A hung build would hold the only worker forever while the heartbeat says it is alive.
        timed_out = threading.Event()

        def _kill() -> None:
            timed_out.set()
            proc.kill()

        watchdog = threading.Timer(EXPORT_TIMEOUT_S, _kill)
        watchdog.daemon = True
        watchdog.start()
        for raw in proc.stdout:
            line = raw.strip()
            output.append(line)
            hit = _phase_of(line)
            if hit is not None and hit[0] != phase:
                step_times[phase] = round(perf_counter() - phase_start, 2)
                phase, phase_start = hit[0], perf_counter()
            if hit is not None:
                task['progress'], task['current_step'] = hit[1], hit[2]
            task['step_times'] = step_times
            store.persist_task(task_id)
        returncode = proc.wait()
        watchdog.cancel()
    step_times[phase] = round(perf_counter() - phase_start, 2)

    if timed_out.is_set():
        raise RuntimeError(f'Export timed out after {EXPORT_TIMEOUT_S:.0f}s and was stopped')
    if returncode != 0:
        tail = '\n'.join(output[-20:])
        raise RuntimeError(f'Export failed (exit {returncode}):\n{tail}')

    task.update(progress=90.0, formats_completed=job['formats'], formats_pending=[])
    store.persist_task(task_id)

    if job.get('auto_load'):
        task.update(status=ExportStatus.LOADING, current_step='Loading into Triton')
        store.persist_task(task_id)
        t0 = perf_counter()
        task['triton_loaded'] = load_into_triton(engines_to_load(job))
        step_times['triton_load'] = round(perf_counter() - t0, 2)

    total = round(perf_counter() - started, 2)
    task.update(
        status=ExportStatus.COMPLETED,
        completed_at=datetime.now(UTC),
        progress=100.0,
        current_step='Complete',
        export_duration_seconds=total,
        step_times=step_times,
        message=f'Model {job["triton_name"]} exported successfully in {total:.1f}s',
    )
    store.persist_task(task_id)
    logger.info('[%s] export completed in %.1fs; step times: %s', task_id, total, step_times)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', type=Path, help='export task dir (default: EXPORT_TASK_DIR)')
    parser.add_argument('--once', action='store_true', help='drain the queue, then exit')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    if args.watch is not None:
        os.environ['EXPORT_TASK_DIR'] = str(args.watch)

    stop = threading.Event()
    beat = threading.Thread(target=_heartbeat_loop, args=(stop,), daemon=True)
    beat.start()
    # `docker compose stop` sends SIGTERM: leave the loop cleanly (a running export
    # finishes or is failed by the next start's reconcile) so the heartbeat can be removed.
    previous = signal.signal(signal.SIGTERM, lambda _sig, _frame: stop.set())
    try:
        failed = reconcile_claimed()
        if failed:
            logger.warning('failed %d export(s) a previous worker left running', failed)
        while not stop.is_set():
            job = claim_next()
            if job is not None:
                run_job(job)
            elif args.once:
                break
            else:
                stop.wait(POLL_INTERVAL_S)
        return 0
    finally:
        stop.set()
        # Joined first so the heartbeat thread cannot rewrite the file after it is removed;
        # without this the API keeps queueing jobs nobody runs until the beat ages out.
        beat.join(HEARTBEAT_INTERVAL_S + 5)
        with contextlib.suppress(OSError):
            store.heartbeat_path().unlink()
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    sys.exit(main())
