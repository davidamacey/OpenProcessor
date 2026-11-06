"""The project registry: an in-process snapshot of every project, refreshed on a cheap poll.

The source of truth is Postgres (``projects``, ``project_indexes``, ``project_slugs``). A poll reads
one integer, ``deployment_counters('projects_revision')``, and loads every project only when it
moved; a status that ends on its own (``building``, ``deleting``) is re-read by slug before a caller
acts on it. The polls use the one-connection poll engine, so they never queue behind requests.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Protocol

from sqlalchemy.orm import Session

from src.core.logging import get_logger
from src.db.engine import get_poll_engine
from src.db.repos import projects as repo
from src.db.session import sync_deployment_tx
from src.services.projects.records import record_from_row


if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.config.projects import ProjectRecord

logger = get_logger(__name__)


class ProjectSource(Protocol):
    """Where a registry reads from. Sync: the registry runs it off the event loop."""

    def read_revision(self) -> int: ...

    def read_all(self) -> tuple[list[ProjectRecord], frozenset[str]]:
        """Every project and the set of retired slugs."""
        ...

    def read_one(self, slug: str) -> ProjectRecord | None: ...


class PostgresProjectSource:
    """The registry's reads over the poll engine, each in its own short deployment transaction."""

    def read_revision(self) -> int:
        with (
            Session(get_poll_engine()) as session,
            sync_deployment_tx(session, 'project registry revision poll'),
        ):
            return repo.read_projects_revision_sync(session)

    def read_all(self) -> tuple[list[ProjectRecord], frozenset[str]]:
        with (
            Session(get_poll_engine()) as session,
            sync_deployment_tx(session, 'project registry snapshot'),
        ):
            records = [record_from_row(r) for r in repo.list_projects_sync(session)]
            return records, frozenset(repo.retired_slugs_sync(session))

    def read_one(self, slug: str) -> ProjectRecord | None:
        with (
            Session(get_poll_engine()) as session,
            sync_deployment_tx(session, 'project registry confirm'),
        ):
            row = repo.get_by_slug_sync(session, slug)
            return record_from_row(row) if row is not None else None


_REFRESH_FAILURE_BACKOFF_SECONDS = 5.0

# Statuses that end on their own; a snapshot saying one is re-read by slug.
_TRANSITIONAL_STATUSES = frozenset({'building', 'deleting'})


class ProjectRegistry:
    """In-process snapshot of every project, kept fresh by :meth:`ensure_fresh` (called from the
    ``bind_path_project`` dependency, so a bind is never more than ~1 s stale) and by
    :meth:`poll_loop`.

    A registry that was never read knows no project at all, so every bind 404s rather than
    guessing."""

    def __init__(self, source: ProjectSource | None = None) -> None:
        self._source: ProjectSource = source if source is not None else PostgresProjectSource()
        self._by_slug: dict[str, ProjectRecord] = {}
        self._retired: frozenset[str] = frozenset()
        self._revision: int = -1
        self._lock = asyncio.Lock()
        self._failed_at: float | None = None
        self._refreshed = False

    @property
    def refreshed(self) -> bool:
        """At least one :meth:`ensure_fresh` read the registry successfully."""
        return self._refreshed

    @property
    def stale(self) -> bool:
        """True right after the most recent :meth:`ensure_fresh` failed: a ``status`` in the
        snapshot may be out of date (e.g. another API instance flipped ``active`` to ``deleting``),
        so a binder must not trust an ``active`` here for writes. Cleared by the next success."""
        return self._failed_at is not None

    def snapshot(self) -> Mapping[str, ProjectRecord]:
        """The last-refreshed view. Cheap, sync, no I/O."""
        return dict(self._by_slug)

    def get(self, slug: str) -> ProjectRecord | None:
        return self._by_slug.get(slug)

    def retired_slugs(self) -> frozenset[str]:
        """Slugs of deleted projects; a retired slug is never reused."""
        return self._retired

    def active_projects(self) -> list[ProjectRecord]:
        """Every ``active`` project, for workers that discover the whole fleet."""
        return [record for record in self.snapshot().values() if record.status == 'active']

    def archived_projects(self) -> list[ProjectRecord]:
        """Every ``archived`` project, for maintenance scripts that still clean up its files."""
        return [record for record in self.snapshot().values() if record.status == 'archived']

    async def ensure_fresh(self) -> None:
        """Poll the revision; reload every project only when it moved.

        An unreachable database keeps the last snapshot (logged, retried after a short backoff);
        an unknown slug still 404s, so nothing fails open."""
        if (
            self._failed_at is not None
            and time.monotonic() - self._failed_at < _REFRESH_FAILURE_BACKOFF_SECONDS
        ):
            return
        try:
            current_revision = await asyncio.to_thread(self._source.read_revision)
            if current_revision != self._revision:
                async with self._lock:
                    if current_revision != self._revision:
                        await self._refresh(current_revision)
            self._failed_at = None
            self._refreshed = True
        except Exception as exc:
            self._failed_at = time.monotonic()
            logger.warning('project_registry_refresh_failed', error=str(exc))

    async def lookup(self, slug: str) -> ProjectRecord | None:
        """The one status read for a slug: every route guard and ``GET /projects/{slug}`` call
        this, so they cannot disagree. A transitional status is confirmed against the row before
        a caller acts on it."""
        await self.ensure_fresh()
        record = self._by_slug.get(slug)
        if record is None or record.status not in _TRANSITIONAL_STATUSES:
            return record
        try:
            confirmed = await asyncio.to_thread(self._source.read_one, slug)
        except Exception as exc:
            logger.warning('project_registry_confirm_failed', slug=slug, error=str(exc))
            return record
        if confirmed is None:
            return record
        if confirmed.status != record.status:
            self._by_slug = {**self._by_slug, slug: confirmed}
        return confirmed

    async def refresh_strict(self) -> None:
        """Reload every project now, raising on any failure. For one-shot scripts and for create,
        where a stale or empty view would skip projects unnoticed."""
        async with self._lock:
            await self._refresh(await asyncio.to_thread(self._source.read_revision))
        self._failed_at = None
        self._refreshed = True

    async def _refresh(self, current_revision: int) -> None:
        records, retired = await asyncio.to_thread(self._source.read_all)
        self._by_slug = {record.slug: record for record in records}
        self._retired = retired
        self._revision = current_revision

    async def poll_loop(self, *, interval_seconds: float = 1.0) -> None:
        """Background refresh loop; started at API startup and cancelled at shutdown.
        :meth:`ensure_fresh` never raises, so a transient database error cannot end the loop."""
        while True:
            await self.ensure_fresh()
            await asyncio.sleep(interval_seconds)


_REGISTRY: ProjectRegistry | None = None


def get_project_registry() -> ProjectRegistry:
    """Process-wide registry singleton, built on first use so tests can swap it first."""
    global _REGISTRY  # noqa: PLW0603 - lazily-built module singleton
    if _REGISTRY is None:
        _REGISTRY = ProjectRegistry()
    return _REGISTRY


def set_project_registry(registry: ProjectRegistry | None) -> None:
    """Test/startup seam."""
    global _REGISTRY  # noqa: PLW0603 - lazily-built module singleton
    _REGISTRY = registry
