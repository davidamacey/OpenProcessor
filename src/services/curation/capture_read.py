"""Read the capture metadata of an ingested file through a small pool of ExifTool processes.

A batch of photos is read at once, so reads run on a few threads, each with its own long-lived
``-stay_open`` process (one process answers one read at a time, which is what its pipe allows).
The photo's bytes reach the process through a named pipe, never through a whole-file temporary
file (:meth:`~src.clients.exiftool_client.ExifToolClient.read_bytes`). An ExifTool outage is
logged by the client and answers :func:`unknown_capture`: the image is ingested without capture,
never with an invented position, and the capture backfill reads it again (it carries no parser
stamp). A read that takes longer than :data:`READ_BOUND_S` in all (waiting for a thread included)
answers the same.

The pool lives per OS process and is closed with the application's lifespan. Its processes are
started by :func:`start_capture_pool` before the process opens a gRPC channel (the API's lifespan,
a job worker's pool child), never lazily between two engine calls
(:mod:`src.clients.exiftool_client`, "without ``preexec_fn``"); a read that finds the process
dead starts a replacement, which is the same safe start.
"""

from __future__ import annotations

import asyncio
import functools
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

from src.clients.exiftool_client import ExifToolClient, ExifToolError, get_exiftool_client
from src.config.project_context import run_in_executor_bound
from src.core.logging import get_logger
from src.domain.capture_metadata import CaptureMetadata, parse_capture, unknown_capture


logger = get_logger(__name__)

#: Most ExifTool processes one API process keeps (each is a Perl interpreter of some tens of MB).
MAX_PROCESSES: Final = 4
MIN_PROCESSES: Final = 2
#: Cores per ExifTool process.
CORES_PER_PROCESS: Final = 8
#: The longest one photo's capture read may take, queue included, before it counts as unknown.
READ_BOUND_S: Final = 60.0


def pool_size(cpus: int) -> int:
    """ExifTool processes for ``cpus`` cores: one per :data:`CORES_PER_PROCESS`, 2 to 4."""
    return min(MAX_PROCESSES, max(MIN_PROCESSES, cpus // CORES_PER_PROCESS))


def _new_client() -> ExifToolClient:
    return ExifToolClient()


class _CapturePool:
    """Worker threads that each own one ExifTool client: one the pool started (:meth:`start`) or,
    when none is left, one started on the thread's first read."""

    def __init__(self, size: int, name: str) -> None:
        self.size = size
        self.executor = ThreadPoolExecutor(max_workers=size, thread_name_prefix=name)
        self._local = threading.local()
        self.lock = threading.Lock()
        self.clients: list[ExifToolClient] = []
        self._unclaimed: list[ExifToolClient] = []

    def start(self) -> None:
        """Start every process now, from the calling thread. A process that will not start is
        logged and left to the thread's first read (which answers unknown capture on an outage)."""
        for _ in range(self.size - len(self.clients)):
            client = _new_client()
            with self.lock:
                self.clients.append(client)
                self._unclaimed.append(client)
            try:
                client.start()
            except ExifToolError as exc:
                logger.warning('exiftool_start_failed', error=str(exc))

    def _client(self) -> ExifToolClient:
        client: ExifToolClient | None = getattr(self._local, 'client', None)
        if client is None:
            with self.lock:
                client = self._unclaimed.pop() if self._unclaimed else None
                if client is None:
                    client = _new_client()
                    self.clients.append(client)
            self._local.client = client
        return client

    def read(self, data: bytes) -> CaptureMetadata:
        try:
            return parse_capture(self._client().read_bytes(data))
        except ExifToolError:
            return unknown_capture()

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self.lock:
            clients, self.clients = self.clients, []
        for client in clients:
            client.close()


@functools.cache
def _pool_of(pid: int) -> _CapturePool:
    """This OS process's pool (keyed by pid: a forked worker never shares its parent's processes)."""
    return _CapturePool(pool_size(os.cpu_count() or 1), f'exiftool-{pid}')


def _pool() -> _CapturePool:
    return _pool_of(os.getpid())


def live_processes() -> list[int]:
    """Process ids of this OS process's running ExifTool processes."""
    if not _pool_of.cache_info().currsize:
        return []
    pool = _pool()
    with pool.lock:
        clients = list(pool.clients)
    return [pid for c in clients if (pid := c.pid()) is not None]


def start_capture_pool() -> None:
    """Start this OS process's ExifTool processes now (see the module doc for when)."""
    _pool().start()


def shutdown_capture_pool() -> None:
    """Stop the threads and close every ExifTool process (application shutdown)."""
    if _pool_of.cache_info().currsize:
        pool = _pool()
        _pool_of.cache_clear()
        pool.close()


def _read_path(path: str) -> CaptureMetadata | None:
    if not Path(path).is_file():
        return None
    return parse_capture(get_exiftool_client().read(path))


async def read_capture(data: bytes) -> CaptureMetadata:
    """The capture metadata of the file whose bytes are ``data``."""
    pool = _pool()
    read = run_in_executor_bound(asyncio.get_running_loop(), pool.executor, pool.read, data)
    try:
        return await asyncio.wait_for(read, READ_BOUND_S)
    except TimeoutError:
        logger.warning('capture_read_timed_out', bound_s=READ_BOUND_S)
        return unknown_capture()


async def read_capture_file(path: str) -> CaptureMetadata | None:
    """The capture metadata of the stored file at ``path``; ``None`` when the file is gone.
    Raises :class:`~src.clients.exiftool_client.ExifToolError` on an ExifTool outage: a backfill
    must stop on one, not stamp every image unknown."""
    return await run_in_executor_bound(
        asyncio.get_running_loop(), _pool().executor, _read_path, path
    )


__all__ = [
    'live_processes',
    'pool_size',
    'read_capture',
    'read_capture_file',
    'shutdown_capture_pool',
    'start_capture_pool',
]
