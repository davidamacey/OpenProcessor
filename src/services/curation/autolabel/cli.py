"""Subprocess entry point for the auto_label pipeline.

Spawned by :func:`src.services.curation.autolabel.job.start_job` via
``python -m src.services.curation.autolabel.cli``. Reads the on-disk
state file produced by ``start_job`` (which contains the pipeline import
path and the call kwargs), constructs an OpenSearch client, and drives
the pipeline to completion.

Lifecycle contract with :mod:`auto_label_job`:

* On launch: read ``state.json`` → resolve pipeline_fn → run it with a
  file-backed progress reporter.
* On clean completion / exception / cancellation: write terminal state
  to ``state.json``, unlink ``running.lock`` and ``cancel.flag``, write
  exit_code to ``/jobs/auto_label/exit_code``.
* SIGTERM (sent by ``cancel_job`` and by the FastAPI lifespan shutdown
  in :func:`auto_label_job.shutdown_active_run`) is translated into
  ``CancelledError`` so the pipeline tears down cleanly via the same
  path the ``cancel.flag`` checkpoint uses.

This module deliberately imports nothing from the FastAPI app surface
beyond what's needed to call the pipeline — fast cold start, no
uvicorn / lifespan side effects.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import signal
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.services.curation.autolabel.job import (
    _CANCEL_FLAG,
    _EXIT_CODE_FILE,
    _RUNNING_LOCK,
    _STATE_FILE,
    _atomic_write,
    _Progress,
    _read_state,
)


def _write_exit_code(code: int) -> None:
    with contextlib.suppress(OSError):
        _EXIT_CODE_FILE.write_text(str(code))


def _resolve_pipeline_fn(pipeline_path: str):
    """``module:qualname`` → callable."""
    mod_name, _, attr_path = pipeline_path.partition(':')
    if not mod_name or not attr_path:
        raise ValueError(f'malformed pipeline path: {pipeline_path!r}')
    mod = importlib.import_module(mod_name)
    obj: Any = mod
    for part in attr_path.split('.'):
        obj = getattr(obj, part)
    return obj


async def _build_opensearch():
    """Construct an OpenSearch client for this subprocess.

    The pipeline expects what ``OpenSearchDep`` resolves to at
    request time — which is the **inner** ``AsyncOpenSearch`` (see
    ``src/routers/curation/_common.py:_raw_opensearch_dep``), not the
    project's ``OpenSearchClient`` wrapper. The wrapper exposes a
    high-level API; curation code talks raw search/bulk/indices.
    """
    from src.core.dependencies import OpenSearchClientFactory

    wrapper = await OpenSearchClientFactory.get_client()
    return getattr(wrapper, 'client', wrapper)


async def _run_pipeline() -> int:
    """Drive the pipeline. Returns an exit code (0 = success, 1 = error,
    130 = cancelled by SIGINT/SIGTERM convention).
    """
    state = _read_state()
    if not state.pipeline:
        msg = 'state.json has no pipeline import path'
        state.status = 'failed'
        state.error = msg
        state.finished_at = time.time()
        _atomic_write(asdict(state))
        sys.stderr.write(f'auto_label_cli: {msg}\n')
        return 1

    try:
        pipeline_fn = _resolve_pipeline_fn(state.pipeline)
    except (ImportError, AttributeError, ValueError) as exc:
        state.status = 'failed'
        state.error = f'cannot resolve pipeline {state.pipeline!r}: {exc}'
        state.finished_at = time.time()
        _atomic_write(asdict(state))
        sys.stderr.write(f'auto_label_cli: {state.error}\n')
        return 1

    try:
        opensearch = await _build_opensearch()
    except Exception as exc:
        state.status = 'failed'
        state.error = f'opensearch init failed: {exc}'
        state.error_detail = traceback.format_exc()[:4096]
        state.finished_at = time.time()
        _atomic_write(asdict(state))
        sys.stderr.write(f'auto_label_cli: {state.error}\n')
        return 1

    progress = _Progress(state)
    kwargs = dict(state.args)
    # The args stored on disk are the serializable subset — opensearch
    # and progress are passed separately below.
    kwargs.pop('opensearch', None)
    kwargs.pop('progress', None)

    rc = 0
    try:
        result = await pipeline_fn(opensearch=opensearch, progress=progress, **kwargs)
        state.result = result if isinstance(result, dict) else {'raw': str(result)}
        state.status = 'completed'
    except asyncio.CancelledError:
        state.status = 'cancelled'
        state.error = state.error or 'cancelled by operator'
        rc = 130
    except BaseException as exc:
        # BLE001: yes we want every non-system exit failure surfaced
        # rather than allowing a numpy/sklearn exception class to escape
        # uncaught. SystemExit / KeyboardInterrupt re-raise below.
        state.status = 'failed'
        state.error = f'{type(exc).__name__}: {exc}'
        state.error_detail = traceback.format_exc()[:4096]
        rc = 1
        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
            state.finished_at = time.time()
            _atomic_write(asdict(state))
            raise
    finally:
        state.finished_at = time.time()
        _atomic_write(asdict(state))
        with contextlib.suppress(FileNotFoundError):
            _RUNNING_LOCK.unlink()
        with contextlib.suppress(FileNotFoundError):
            _CANCEL_FLAG.unlink()
    return rc


async def _amain() -> int:
    """Run the pipeline; install signal handlers that cancel it on
    SIGTERM/SIGINT (sent by cancel_job and by the API lifespan shutdown).
    The CancelledError unwinds through ``_run_pipeline`` which writes
    terminal state to disk.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(_run_pipeline())

    def _cancel() -> None:
        if not task.done():
            task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, ValueError):
            loop.add_signal_handler(sig, _cancel)

    try:
        return await task
    except asyncio.CancelledError:
        return 130


def main() -> int:
    # Ensure the state directory exists (it should already; defensive).
    Path(_STATE_FILE.parent).mkdir(parents=True, exist_ok=True)
    try:
        rc = asyncio.run(_amain())
    except KeyboardInterrupt:
        rc = 130
    except BaseException as exc:
        # Catch-all so we still write exit_code even on weird failures.
        sys.stderr.write(f'auto_label_cli: fatal: {type(exc).__name__}: {exc}\n')
        with contextlib.suppress(Exception):
            state = _read_state()
            state.status = 'failed'
            state.error = state.error or f'fatal: {type(exc).__name__}: {exc}'
            state.error_detail = traceback.format_exc()[:4096]
            state.finished_at = time.time()
            _atomic_write(asdict(state))
        rc = 1
    _write_exit_code(rc)
    return rc


if __name__ == '__main__':
    sys.exit(main())
