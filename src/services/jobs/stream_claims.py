"""A stream-worker replica claims whole projects and keeps its fair share (02b section 5).

Every pass (the producer runs one every ``HEARTBEAT_TICK_S``):

1. the replica upserts its ``worker_capabilities`` row, so a replica with no claim yet counts;
2. ``live`` is the replicas of its kind seen within the lease TTL, itself included, and
   ``fair = ceil(active_projects / live)``;
3. it renews the claims it holds (those of projects no longer active or now paused are dropped);
4. over its share, it releases the excess, oldest claim first, at a project boundary (only
   projects the caller reports idle), so a newly started replica gets work within two passes;
5. under its share, it claims unclaimed or expired projects, skipping paused ones;
6. on shutdown it releases every claim and its liveness row.

One replica gets ``fair = active``: it behaves as the round-robin over every project did. A
pass that cannot reach the database keeps the claims already held until the local fence
(``LEASE_TTL_S - FENCE_MARGIN_S`` after the last good pass), then holds nothing: work stops
rather than run unclaimed (fail closed).
"""

from __future__ import annotations

import asyncio
import logging
import math
import socket
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.db.engine import get_sync_engine
from src.db.repos import stream_claims as repo
from src.db.session import sync_deployment_tx
from src.domain import job_timing


if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable, Collection, Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """An active project the caller offers for claiming."""

    project_id: uuid.UUID
    slug: str
    paused: bool = False


@dataclass
class ProjectClaims:
    """The claims of one replica of one stream worker (``detection`` or ``vlm``)."""

    kind: str
    ttl_s: float = job_timing.LEASE_TTL_S
    pass_every_s: float = job_timing.HEARTBEAT_TICK_S
    margin_s: float = job_timing.FENCE_MARGIN_S
    host: str = field(default_factory=socket.gethostname)
    clock: Callable[[], float] = time.monotonic
    session_factory: Callable[[], Session] | None = None
    held: dict[uuid.UUID, int] = field(default_factory=dict)
    _acquired_at: dict[uuid.UUID, float] = field(default_factory=dict)
    _last_good: float | None = None
    _last_pass: float | None = None

    @property
    def me(self) -> str:
        return repo.worker_id(self.kind, self.host)

    def _session(self) -> Session:
        if self.session_factory is not None:
            return self.session_factory()
        return Session(get_sync_engine(), expire_on_commit=False)

    # --- the pass ---------------------------------------------------------------------------

    def run_pass(
        self, candidates: Iterable[Candidate], idle_slugs: Iterable[str] = ()
    ) -> dict[uuid.UUID, int]:
        """One pass against the database. Returns ``{project: fencing token}`` now held."""
        offered = {c.project_id: c for c in candidates if not c.paused}
        idle = set(idle_slugs)
        now = self.clock()
        with self._session() as session, sync_deployment_tx(session, 'stream worker claims'):
            repo.touch_worker(session, self.kind, self.me, self.host)
            live = max(1, repo.live_workers(session, self.kind, self.ttl_s))
            fair = math.ceil(len(offered) / live) if offered else 0
            held = repo.renew_held(session, self.kind, self.me, self.ttl_s)
            for project in [p for p in held if p not in offered]:
                repo.release(session, self.kind, project, self.me)  # archived, paused or deleted
                del held[project]
            for project in held:
                self._acquired_at.setdefault(project, 0.0)  # held from before this process knew
            if len(held) > fair:
                oldest_first = sorted(held, key=lambda p: self._acquired_at.get(p, 0.0))
                for project in oldest_first:
                    if len(held) <= fair:
                        break
                    if offered[project].slug in idle:
                        repo.release(session, self.kind, project, self.me)
                        del held[project]
            if len(held) < fair:
                for project in offered:
                    if len(held) >= fair:
                        break
                    if project in held:
                        continue
                    token = repo.claim(session, self.kind, project, self.me, self.ttl_s)
                    if token is not None:
                        held[project] = token
                        self._acquired_at[project] = now
        for project in [p for p in self._acquired_at if p not in held]:
            del self._acquired_at[project]
        self.held = held
        self._last_good = now
        return dict(held)

    # --- the producer's side ----------------------------------------------------------------

    async def refresh(
        self, candidates: Iterable[Candidate], *, busy: Collection[str] = ()
    ) -> set[uuid.UUID]:
        """The projects this replica may serve now. Runs a pass at most every ``pass_every_s``.

        ``busy`` names the projects with work in flight: a share above the fair one is given
        back only from the others, at a project boundary."""
        offered = list(candidates)
        now = self.clock()
        if self._last_pass is not None and now - self._last_pass < self.pass_every_s:
            return self._valid_held(now)
        self._last_pass = now
        idle_slugs = [c.slug for c in offered if c.slug not in busy]
        try:
            held = await asyncio.to_thread(self.run_pass, offered, idle_slugs)
        except (SQLAlchemyError, OSError) as exc:
            logger.warning('stream_claims_pass_failed kind=%s error=%s', self.kind, exc)
            return self._valid_held(now)
        return set(held)

    async def select(self, records: Iterable[Any], *, busy: Collection[str] = ()) -> list[Any]:
        """The project records this replica serves now, in the order given.

        A record without a database id cannot be claimed and is never served (fail closed).
        """
        records = list(records)
        offered = [
            Candidate(r.id, r.slug, paused=r.paused_at is not None)
            for r in records
            if r.id is not None
        ]
        held = await self.refresh(offered, busy=busy)
        return [r for r in records if r.id in held]

    def _valid_held(self, now: float) -> set[uuid.UUID]:
        """The claims already held, while the last good pass is young enough to trust them."""
        if self._last_good is None or now - self._last_good >= self.ttl_s - self.margin_s:
            return set()
        return set(self.held)

    def release_all(self) -> int:
        """Shutdown: give every project back and stop counting toward the others' share."""
        try:
            with self._session() as session, sync_deployment_tx(session, 'stream worker release'):
                count = repo.release_all(session, self.kind, self.me)
                repo.forget_worker(session, self.me)
        except (SQLAlchemyError, OSError) as exc:
            logger.warning('stream_claims_release_failed kind=%s error=%s', self.kind, exc)
            return 0
        self.held = {}
        self._acquired_at.clear()
        return count

    async def arelease_all(self) -> int:
        return await asyncio.to_thread(self.release_all)
