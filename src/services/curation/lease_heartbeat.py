"""The heartbeat thread of a held job lease.

A lease (:mod:`src.clients.job_lease`) stays alive only while its holder renews it. The
renew runs on a ``threading.Thread(daemon=True)`` with its own event loop and its own
OpenSearch client, never on the request loop: CPU work on that loop (an upload extract, a
``k_center_greedy`` selection) used to stall an asyncio ticker and cost a live job its
liveness (Python docs: https://docs.python.org/3/library/threading.html,
https://docs.python.org/3/library/asyncio-runner.html -- one loop per thread, asyncio objects
are not thread-safe). The bound project reaches the thread through a copied
``contextvars`` context (https://docs.python.org/3/library/contextvars.html).

The thread also touches the job's file heartbeat, which is what the UI shows. When a renew
answers :class:`~src.domain.job_lease.Lost`, or no renew has succeeded for
``LEASE_TTL_S - FENCE_MARGIN_S`` seconds, it sets :attr:`LeaseHeartbeat.lost`: the holder
stops at its next progress write before anyone can see the lease as expired.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
import time
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients import job_lease
from src.config.settings import get_settings
from src.core.logging import get_logger
from src.domain.job_lease import (
    FENCE_MARGIN_S,
    HEARTBEAT_TICK_S,
    LEASE_TTL_S,
    Busy,
    HeldLease,
    LeaseKind,
    Lost,
    Renewed,
)
from src.services.curation.file_job import clear_fence, register_fence


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from src.services.curation.file_job import FileJob

logger = get_logger(__name__)


def new_lease_client() -> Any:
    """A guarded OpenSearch client of its own (the caller closes it): for the heartbeat
    thread and for a short lease taken where no request client is at hand."""
    from src.services.projects.guard import make_script_opensearch

    settings = get_settings()
    return make_script_opensearch([settings.opensearch_url], timeout=settings.opensearch_timeout)


class LeaseHeartbeat:
    """Renews ``lease`` every :data:`HEARTBEAT_TICK_S` until :meth:`stop`."""

    def __init__(
        self,
        lease: HeldLease,
        *,
        job: FileJob | None = None,
        client_factory: Callable[[], Any] = new_lease_client,
    ) -> None:
        self.lease = lease
        self.lost = threading.Event()
        self.last_renew_ok = time.monotonic()
        self._job = job
        self._client_factory = client_factory
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._job is not None:
            self._job.touch_heartbeat()
            register_fence(self._job.directory, self.may_write)
        ctx = contextvars.copy_context()
        self._thread = threading.Thread(
            target=ctx.run, args=(self._thread_main,), name='lease-heartbeat', daemon=True
        )
        self._thread.start()

    def may_write(self) -> bool:
        """Whether a progress write is still safe: not lost, and the last renew is young
        enough that no other process can yet see the lease as expired."""
        if self.lost.is_set():
            return False
        return time.monotonic() - self.last_renew_ok < LEASE_TTL_S - FENCE_MARGIN_S

    def stop(self) -> None:
        """Stop renewing and wait for an in-flight renew to finish (at most one round trip)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=HEARTBEAT_TICK_S)
        if self._job is not None:
            clear_fence(self._job.directory)

    async def confirm(self, client: Any) -> bool:
        """Stop the thread, then renew once on ``client``: ``True`` only when the lease is
        still ours. A terminal state is written only after this returns ``True``."""
        await asyncio.to_thread(self.stop)
        if self.lost.is_set():
            return False
        result = await job_lease.renew(client, self.lease)
        if isinstance(result, Renewed):
            self.lease = result.lease
            self.last_renew_ok = time.monotonic()
            return True
        self.lost.set()
        return False

    async def release(self, client: Any) -> None:
        """Stop renewing and release the lease (a lease taken over meanwhile is left alone)."""
        await asyncio.to_thread(self.stop)
        await job_lease.release(client, self.lease)

    def _thread_main(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        client = self._client_factory()
        try:
            while not self._stop.wait(HEARTBEAT_TICK_S):
                if self._job is not None:
                    self._job.touch_heartbeat()
                try:
                    result = await job_lease.renew(client, self.lease)
                except (OpenSearchException, OSError) as exc:
                    logger.warning('lease_renew_failed', lease=self.lease.doc_id, error=str(exc))
                    if time.monotonic() - self.last_renew_ok >= LEASE_TTL_S - FENCE_MARGIN_S:
                        self.lost.set()
                        return
                    continue
                if isinstance(result, Lost):
                    logger.warning('lease_lost', lease=self.lease.doc_id)
                    self.lost.set()
                    return
                self.lease = result.lease
                self.last_renew_ok = time.monotonic()
        finally:
            await client.close()


async def acquire_heartbeat(
    client: Any,
    kind: LeaseKind,
    scope: str | None,
    job_id: str,
    *,
    job: FileJob | None,
    busy: Callable[[str | None], Exception],
) -> LeaseHeartbeat:
    """Take the lease of a job about to start; the caller starts the heartbeat
    (:meth:`LeaseHeartbeat.start`) once the job's first state is written.

    Check the family's record-based busy test *before* this call: it is what keeps a
    pre-lease job (no lease document, fresh heartbeat) busy. Raises ``busy(holder_job_id)`` -- the family's own busy error -- when a live holder
    exists; any OpenSearch error propagates, so a start with OpenSearch down refuses
    instead of proceeding without a lease. The caller releases the lease
    (:meth:`LeaseHeartbeat.release`) when its start step fails; :func:`run_leased` releases
    it when the run ends.
    """
    result = await job_lease.acquire(client, kind, scope, job_id)
    if isinstance(result, Busy):
        raise busy(result.job_id)
    return LeaseHeartbeat(result.lease, job=job)


_HELD: dict[str, LeaseHeartbeat] = {}


def hold(key: str, beat: LeaseHeartbeat) -> None:
    """Remember the lease this process took for ``key`` between a claim and the spawn of
    its run (a claim and a spawn are separate steps of one request)."""
    _HELD[key] = beat


def take(key: str) -> LeaseHeartbeat:
    """The lease :func:`hold` remembered for ``key``; ``KeyError`` when the claim was never
    made in this process (a run must not start without its lease)."""
    return _HELD.pop(key)


async def run_leased(beat: LeaseHeartbeat, client: Any, run: Awaitable[None]) -> None:
    """Await ``run``, then release the lease (also when it fails or is cancelled)."""
    try:
        await run
    finally:
        await beat.release(client)


@contextlib.asynccontextmanager
async def leased(
    client: Any,
    kind: LeaseKind,
    scope: str | None,
    job_id: str,
    *,
    busy: Callable[[str | None], Exception],
) -> AsyncIterator[LeaseHeartbeat]:
    """Hold a lease, with its heartbeat thread, for the body of an ``async with`` (a short
    write that finishes inside one request); released on exit, also on an error."""
    beat = await acquire_heartbeat(client, kind, scope, job_id, job=None, busy=busy)
    beat.start()
    try:
        yield beat
    finally:
        await beat.release(client)
