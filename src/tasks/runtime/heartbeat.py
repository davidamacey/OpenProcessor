"""The heartbeat thread of a running job (02b section 3.1, step 3).

A job holds a lease on its ``jobs`` row; this thread renews it every ``HEARTBEAT_TICK_S`` on a
sync connection of its own, never the task's, and never from the task's event loop (CPU work on
that loop used to starve an asyncio ticker and cost a live job its liveness; Python threading and
asyncio docs: one loop per thread, asyncio objects are not thread-safe). The renew is fenced by
the claim's token: zero rows is ``Lost`` and sets the stop event. The returned cancel and pause
requests set their own events, which the handler reads through ``ctx.cancelled()`` and
``ctx.paused()``.

``may_write()`` is the self-fence for writes outside the database (OpenSearch): false once the
last good renew is older than ``LEASE_TTL_S - FENCE_MARGIN_S``, so a holder that cannot reach the
database stops before a reaper can take the job over. Every beat also refreshes the GPU card
inventory of the process.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from typing import TYPE_CHECKING

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.db.engine import get_sync_engine
from src.db.repos import jobs as repo
from src.domain import job_timing
from src.tasks.runtime import gpu_app
from src.tasks.runtime.scoped import job_transaction


if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class JobHeartbeat:
    """Renews the lease of one claimed job until :meth:`stop`."""

    def __init__(
        self,
        job_id: uuid.UUID,
        project_id: uuid.UUID | None,
        token: int,
        *,
        tick_s: float | None = None,
        ttl_s: float | None = None,
        fence_margin_s: float | None = None,
        session_factory: Callable[[], Session] | None = None,
    ) -> None:
        self.job_id = job_id
        self.project_id = project_id
        self.token = token
        self._tick_s = tick_s if tick_s is not None else job_timing.HEARTBEAT_TICK_S
        self._ttl_s = ttl_s if ttl_s is not None else job_timing.LEASE_TTL_S
        self._margin_s = fence_margin_s if fence_margin_s is not None else job_timing.FENCE_MARGIN_S
        self._session_factory = session_factory or _default_session
        self.lost = threading.Event()
        self.cancel_requested = threading.Event()
        self.pause_requested = threading.Event()
        self._last_good = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        ctx = contextvars.copy_context()
        self._thread = threading.Thread(
            target=ctx.run, args=(self._run,), name=f'job-heartbeat-{self.job_id}', daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop renewing and wait for an in-flight renew (at most one round trip)."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(self._tick_s, 1.0))

    def may_write(self) -> bool:
        """Whether a write outside the database is still safe: not lost, last renew young."""
        if self.lost.is_set():
            return False
        return time.monotonic() - self._last_good < self._ttl_s - self._margin_s

    def note_control(self, cancel_requested: bool, pause_requested: bool) -> None:
        """A fenced progress or checkpoint write learned of a request before the next beat."""
        if cancel_requested:
            self.cancel_requested.set()
        if pause_requested:
            self.pause_requested.set()

    def _run(self) -> None:
        while not self._stop.wait(self._tick_s):
            if not self._beat():
                return

    def _beat(self) -> bool:
        """One renew. ``False`` ends the thread (the lease is lost)."""
        try:
            with self._session_factory() as session, job_transaction(session, self.project_id):
                control = repo.heartbeat(session, self.job_id, self.token, int(self._ttl_s))
        except (SQLAlchemyError, OSError) as exc:
            logger.warning('job_heartbeat_failed job_id=%s error=%s', self.job_id, exc)
            if time.monotonic() - self._last_good >= self._ttl_s - self._margin_s:
                self.lost.set()
                return False
            return True
        if control is None:
            logger.warning('lease_lost job_id=%s token=%s', self.job_id, self.token)
            self.lost.set()
            return False
        self._last_good = time.monotonic()
        self.note_control(control.cancel_requested, control.pause_requested)
        self._refresh_inventory()
        return True

    def _refresh_inventory(self) -> None:
        try:
            gpu_app.refresh_inventory()
        except (SQLAlchemyError, OSError) as exc:
            logger.warning('gpu_inventory_refresh_failed error=%s', exc)


def _default_session() -> Session:
    return Session(get_sync_engine(), expire_on_commit=False)
