"""Per-process thread pool and admission gate for per-image CPU work.

Decode, letterbox, PE preprocessing, blur scoring and crop JPEG encoding are
numpy, OpenCV and Pillow calls that release the GIL. Run directly in a
coroutine they hold the event-loop thread, so one worker cannot overlap two
images and a request waiting on the loop looks like a slow datastore call.
:func:`run_cpu` moves such a call to a small pool owned by the process, with
the caller's ``contextvars`` (the bound project, the request id) copied
across the hop.

:class:`InflightGate` bounds how many images a process holds decoded at once.
Past the cap a request queues (first in, first out) before it decodes rather
than all requests decoding together, so more clients cannot lower the
throughput and the resident set stays bounded.
"""

from __future__ import annotations

import asyncio
import functools
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from src.config.project_context import run_in_executor_bound


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


DEFAULT_CPU_THREADS = 4
DEFAULT_INFLIGHT_IMAGES = 4

_executor: ThreadPoolExecutor | None = None
_executor_pid = 0
_executor_lock = threading.Lock()


def cpu_threads() -> int:
    """Threads in this process's CPU pool (``OP_CPU_POOL_THREADS``)."""
    raw = os.environ.get('OP_CPU_POOL_THREADS', '').strip()
    return max(1, int(raw) if raw else DEFAULT_CPU_THREADS)


def inflight_images() -> int:
    """Images one process holds in the ingest pipeline at once (``OP_INGEST_INFLIGHT_IMAGES``)."""
    raw = os.environ.get('OP_INGEST_INFLIGHT_IMAGES', '').strip()
    return max(1, int(raw) if raw else DEFAULT_INFLIGHT_IMAGES)


def cpu_executor() -> ThreadPoolExecutor:
    """The process's CPU pool, created on first use (and again after a fork)."""
    global _executor, _executor_pid
    pid = os.getpid()
    with _executor_lock:
        if _executor is None or _executor_pid != pid:
            _executor = ThreadPoolExecutor(
                max_workers=cpu_threads(), thread_name_prefix='op-cpu-'
            )
            _executor_pid = pid
        return _executor


def shutdown_cpu_executor() -> None:
    """Stop the pool at application shutdown (running calls finish first)."""
    global _executor
    with _executor_lock:
        executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown(wait=True)


async def run_cpu[T](fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run ``fn(*args, **kwargs)`` on the CPU pool and await the result."""
    call = functools.partial(fn, **kwargs) if kwargs else fn
    return await run_in_executor_bound(asyncio.get_running_loop(), cpu_executor(), call, *args)


class InflightGate:
    """A first-in first-out gate that admits at most ``capacity`` units at once.

    A request holds as many units as it has images. One that wants more than
    the whole capacity is clamped to it, so it still runs, alone, instead of
    waiting forever.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f'capacity must be at least 1, got {capacity}')
        self.capacity = capacity
        self._free = capacity
        self._waiters: deque[tuple[int, asyncio.Future[None]]] = deque()

    @property
    def queued(self) -> int:
        return len(self._waiters)

    @asynccontextmanager
    async def hold(self, units: int) -> AsyncIterator[None]:
        units = min(max(1, units), self.capacity)
        await self._acquire(units)
        try:
            yield
        finally:
            self._free += units
            self._wake()

    async def _acquire(self, units: int) -> None:
        if not self._waiters and self._free >= units:
            self._free -= units
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (units, future)
        self._waiters.append(entry)
        try:
            await future
        except BaseException:
            if future.done() and not future.cancelled():
                # Admitted at the moment the wait was cancelled: hand the units back.
                self._free += units
            elif entry in self._waiters:
                self._waiters.remove(entry)
            self._wake()
            raise

    def _wake(self) -> None:
        while self._waiters and self._waiters[0][0] <= self._free:
            units, future = self._waiters.popleft()
            if future.cancelled():
                continue
            self._free -= units
            future.set_result(None)


_gate: InflightGate | None = None
_gate_pid = 0


def ingest_gate() -> InflightGate:
    """The process's ingest admission gate, created on first use (and again after a fork)."""
    global _gate, _gate_pid
    pid = os.getpid()
    if _gate is None or _gate_pid != pid:
        _gate = InflightGate(inflight_images())
        _gate_pid = pid
    return _gate


__all__ = [
    'DEFAULT_CPU_THREADS',
    'DEFAULT_INFLIGHT_IMAGES',
    'InflightGate',
    'cpu_executor',
    'cpu_threads',
    'ingest_gate',
    'inflight_images',
    'run_cpu',
    'shutdown_cpu_executor',
]
