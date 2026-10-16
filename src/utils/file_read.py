"""Read a batch's source files in parallel on a small per-process thread pool.

The files of a bulk run sit on a network share, where one 7 MB read costs tens of milliseconds
and sixteen of them one after another cost a second or more. Read on the event loop thread
they also stall everything else the worker awaits, which shows up as inflated OpenSearch and
Triton stage times. A separate pool keeps those reads from queueing behind decode work, and
its size bounds the reads in flight per process (``OP_INGEST_READ_THREADS``).
"""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.config.env import env_int
from src.config.project_context import run_in_executor_bound
from src.utils.stage_timing import add_stage_bytes, stage_timer


DEFAULT_READ_THREADS = 8

_executors: dict[int, ThreadPoolExecutor] = {}
_lock = threading.Lock()


def read_threads() -> int:
    """Threads in this process's file-read pool (``OP_INGEST_READ_THREADS``)."""
    return env_int('OP_INGEST_READ_THREADS', DEFAULT_READ_THREADS, minimum=1)


def _read_executor() -> ThreadPoolExecutor:
    pid = os.getpid()
    with _lock:
        if pid not in _executors:
            _executors[pid] = ThreadPoolExecutor(
                max_workers=read_threads(), thread_name_prefix='op-read-'
            )
        return _executors[pid]


def shutdown_read_executor() -> None:
    """Stop the pool (application shutdown, or a test changing its size)."""
    with _lock:
        executor = _executors.pop(os.getpid(), None)
    if executor is not None:
        executor.shutdown(wait=True)


def _read_one(path: str) -> bytes:
    return Path(path).read_bytes()


async def read_files(paths: list[str]) -> list[bytes | OSError]:
    """The bytes of each path in order; a file that cannot be read is its ``OSError``."""
    loop = asyncio.get_running_loop()
    executor = _read_executor()

    async def one(path: str) -> bytes | OSError:
        try:
            return await run_in_executor_bound(loop, executor, _read_one, path)
        except OSError as exc:
            return exc

    with stage_timer('file_read'):
        out = list(await asyncio.gather(*(one(p) for p in paths)))
    add_stage_bytes('file_read', sum(len(r) for r in out if isinstance(r, bytes)))
    return out


__all__ = ['DEFAULT_READ_THREADS', 'read_files', 'read_threads', 'shutdown_read_executor']
