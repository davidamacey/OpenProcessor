"""The ``op_projects`` registry: one doc per project plus a revision
counter, and an in-process snapshot refreshed on a cheap poll (one GET of
the counter; a ``_search`` only when it changed).

See ``docs/design/openprocessor_internal/projects_plan.md`` §4.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.projects import ProjectRecord, ProjectResources
from src.core.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Mapping

logger = get_logger(__name__)

REVISION_DOC_ID = 'meta:projects_revision'


def projects_index() -> str:
    return os.environ.get('OP_PROJECTS_INDEX', 'op_projects')


def _project_doc_id(slug: str) -> str:
    return f'project:{slug}'


def _resources_to_dict(resources: ProjectResources) -> dict[str, Any]:
    return {
        'indexes': {role.value: name for role, name in resources.indexes.items()},
        'class_registry_path': str(resources.class_registry_path),
        'export_root': str(resources.export_root),
        'upload_root': str(resources.upload_root),
        'bakeoff_eval_root': str(resources.bakeoff_eval_root),
        'project_state_dir': str(resources.project_state_dir),
        'train_jobs_dir': str(resources.train_jobs_dir),
        'autolabel_dir': str(resources.autolabel_dir),
        'bakeoff_jobs_dir': str(resources.bakeoff_jobs_dir),
        'mlflow_experiment': resources.mlflow_experiment,
        'model_prefix': resources.model_prefix,
    }


def _resources_from_dict(data: Mapping[str, Any]) -> ProjectResources:
    from src.config.curation import IndexRole

    return ProjectResources(
        indexes={IndexRole(role): name for role, name in data['indexes'].items()},
        class_registry_path=Path(data['class_registry_path']),
        export_root=Path(data['export_root']),
        upload_root=Path(data['upload_root']),
        bakeoff_eval_root=Path(data['bakeoff_eval_root']),
        project_state_dir=Path(data['project_state_dir']),
        train_jobs_dir=Path(data['train_jobs_dir']),
        autolabel_dir=Path(data['autolabel_dir']),
        bakeoff_jobs_dir=Path(data['bakeoff_jobs_dir']),
        mlflow_experiment=data['mlflow_experiment'],
        model_prefix=data['model_prefix'],
    )


def record_to_doc(record: ProjectRecord) -> dict[str, Any]:
    return {
        'slug': record.slug,
        'display_name': record.display_name,
        'description': record.description,
        'status': record.status,
        'revision': record.revision,
        'created_at': record.created_at,
        'updated_at': record.updated_at,
        'origin': record.origin,
        'resources': _resources_to_dict(record.resources),
        'pre_delete_status': record.pre_delete_status,
    }


def doc_to_record(doc: Mapping[str, Any]) -> ProjectRecord:
    return ProjectRecord(
        slug=doc['slug'],
        display_name=doc['display_name'],
        description=doc['description'],
        status=doc['status'],
        revision=doc['revision'],
        created_at=doc['created_at'],
        updated_at=doc['updated_at'],
        origin=doc.get('origin'),
        resources=_resources_from_dict(doc['resources']),
        pre_delete_status=doc.get('pre_delete_status'),
    )


async def get_record_with_seq(
    client: Any, slug: str
) -> tuple[ProjectRecord | None, int | None, int | None]:
    """The stored record plus its ``_seq_no``/``_primary_term``, for an
    OCC-guarded write. ``None`` (with no seq/term) when the doc does not
    exist yet."""
    try:
        doc = await client.get(index=projects_index(), id=_project_doc_id(slug))
    except Exception as exc:
        if getattr(exc, 'status_code', None) == 404 or 'NotFound' in type(exc).__name__:
            return None, None, None
        raise
    if not doc.get('found', True):
        return None, None, None
    return doc_to_record(doc['_source']), doc.get('_seq_no'), doc.get('_primary_term')


class RevisionConflictError(Exception):
    """Raised by :func:`write_record` when the ``if_seq_no``/
    ``if_primary_term`` OCC guard lost a race against another writer --
    i.e. a second bump landed between this caller's read and its write.
    Callers translate this into the API's 409 ``revision_conflict``
    (never a silent overwrite; never a bare 500)."""


def _is_conflict_exception(exc: Exception) -> bool:
    if getattr(exc, 'status_code', None) == 409:
        return True
    return 'Conflict' in type(exc).__name__


async def write_record(
    client: Any,
    record: ProjectRecord,
    *,
    if_seq_no: int | None = None,
    if_primary_term: int | None = None,
    op_type: str | None = None,
) -> None:
    """Write ``record`` (create or OCC-guarded overwrite) and bump the
    registry revision. Raises :class:`RevisionConflictError` when
    ``if_seq_no``/``if_primary_term`` are stale -- i.e. another writer's
    bump landed first (the storage-level race this guards against;
    callers translate it into the API's 409 ``revision_conflict``).
    ``op_type='create'`` makes the write itself refuse a doc that already
    exists (the storage-level create race two concurrent ``POST
    /projects`` for the same slug would otherwise lose -- M1)."""
    from src.services.projects.bootstrap import bump_revision
    from src.services.projects.guard import bind_registry_admin

    kwargs: dict[str, Any] = {}
    if if_seq_no is not None:
        kwargs['if_seq_no'] = if_seq_no
    if if_primary_term is not None:
        kwargs['if_primary_term'] = if_primary_term
    if op_type is not None:
        kwargs['op_type'] = op_type
    # The guard only lets lifecycle code write op_projects; a create has no
    # project bound yet, so every registry write declares itself here.
    with bind_registry_admin():
        try:
            await client.index(
                index=projects_index(),
                id=_project_doc_id(record.slug),
                body=record_to_doc(record),
                # B2: op_projects is tiny -- read-your-writes here, not the
                # ~1s default refresh interval. Without this, ensure_fresh's
                # _search (which the guard maps index -> project from) can
                # still see the pre-write snapshot even after this index()
                # call returns, so a just-created project's own
                # op_prj_<slug>__* index creation gets refused
                # "belongs to no known project" moments after create_project
                # wrote the doc.
                refresh='wait_for',
                # B2: op_projects is tiny -- read-your-writes here, not the
                # ~1s default refresh interval. Without this, ensure_fresh's
                # _search (which the guard maps index -> project from) can
                # still see the pre-write snapshot even after this index()
                # call returns, so a just-created project's own
                # op_prj_<slug>__* index creation gets refused
                # "belongs to no known project" moments after create_project
                # wrote the doc.
                **kwargs,
            )
        except Exception as exc:
            if _is_conflict_exception(exc):
                raise RevisionConflictError(f'revision conflict writing {record.slug!r}') from exc
            raise
        await bump_revision(client)


async def _read_revision(client: Any) -> int:
    """The ``meta:projects_revision`` counter; 0 when the doc (or the whole
    index) does not exist yet. Any other failure propagates."""
    try:
        counter_doc = await client.get(index=projects_index(), id=REVISION_DOC_ID)
    except Exception as exc:
        if getattr(exc, 'status_code', None) == 404 or 'NotFound' in type(exc).__name__:
            return 0
        raise
    return int((counter_doc.get('_source') or {}).get('revision', 0))


_REFRESH_PAGE_SIZE = 500

# After a failed refresh, request-path callers skip OpenSearch for this
# long instead of paying a connection error on every bind.
_REFRESH_FAILURE_BACKOFF_SECONDS = 5.0


class ProjectRegistry:
    """In-process snapshot of every project record, kept fresh by
    :meth:`ensure_fresh` (called from the ``bind_path_project`` dependency,
    so a bind is never more than ~1s stale) and by :meth:`poll_loop` (a
    background task started at API startup).

    Every project, ``default`` included, is exactly its stored record:
    nothing is synthesized, so a registry that was never read knows no
    project at all (and every bind 404s rather than guessing)."""

    def __init__(self, client_factory: Any) -> None:
        """``client_factory`` is a zero-arg callable (sync or async)
        returning the raw OpenSearch client to query."""
        self._client_factory = client_factory
        self._by_slug: dict[str, ProjectRecord] = {}
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
        """True right after the most recent :meth:`ensure_fresh` failed
        (P1R minor 10): the snapshot's ``status`` for any project may be
        out of date -- e.g. a project flipped ``active`` -> ``deleting``
        by another API instance between this instance's last successful
        refresh and now. A binder that trusts a stale ``active`` here
        would let writes through against a project mid-delete. Cleared
        by the next successful refresh."""
        return self._failed_at is not None

    def snapshot(self) -> Mapping[str, ProjectRecord]:
        """The last-refreshed view. Cheap, sync, no I/O -- callers that
        need at-most-1s staleness should call :meth:`ensure_fresh` first."""
        return dict(self._by_slug)

    def get(self, slug: str) -> ProjectRecord | None:
        return self._by_slug.get(slug)

    def active_projects(self) -> list[ProjectRecord]:
        """Every ``active`` project (``default`` included), for workers
        that must discover the whole fleet instead of binding one slug.
        Archived/deleting/building/failed projects are excluded -- a
        worker skips them entirely, the same way a request to their
        indexes would 404/409 at the route layer."""
        return [record for record in self.snapshot().values() if record.status == 'active']

    def existing_projects(self) -> list[ProjectRecord]:
        """Every project that exists and is not being deleted (any status but
        ``deleting``/``deleted``): the one set a cross-project "who uses X"
        listing may count. A tombstone keeps its record but its indexes are
        gone, so counting it over-reports (and blocks deletes as ``in_use``)."""
        return [
            record
            for record in self.snapshot().values()
            if record.status not in ('deleting', 'deleted')
        ]

    def archived_projects(self) -> list[ProjectRecord]:
        """Every ``archived`` project, for maintenance scripts
        (``prune_exports.py``, ``prune_training_runs.py``) that must still
        clean up a project's own files after it stops taking traffic."""
        return [record for record in self.snapshot().values() if record.status == 'archived']

    async def ensure_fresh(self) -> None:
        """One GET of the revision counter; a ``_search`` over every
        project doc only when the counter moved.

        An unreachable registry keeps the last snapshot (logged, then
        retried after a short backoff); an unknown slug still 404s, so
        nothing fails open."""
        if (
            self._failed_at is not None
            and time.monotonic() - self._failed_at < _REFRESH_FAILURE_BACKOFF_SECONDS
        ):
            return
        try:
            client = self._client_factory()
            if asyncio.iscoroutine(client):
                client = await client
            current_revision = await _read_revision(client)
            if current_revision == self._revision:
                self._failed_at = None
                self._refreshed = True
                return
            async with self._lock:
                if current_revision != self._revision:
                    await self._refresh(client, current_revision)
            self._failed_at = None
            self._refreshed = True
        except Exception as exc:
            self._failed_at = time.monotonic()
            logger.warning('project_registry_refresh_failed', error=str(exc))

    async def refresh_strict(self) -> None:
        """Reload every project doc now, raising on any failure. For
        one-shot maintenance scripts, where silently falling back to a
        stale or ``default``-only view would skip projects unnoticed."""
        client = self._client_factory()
        if asyncio.iscoroutine(client):
            client = await client
        async with self._lock:
            await self._refresh(client, await _read_revision(client))
        self._failed_at = None

    async def _refresh(self, client: Any, current_revision: int) -> None:
        """Read every project doc, a page at a time (``search_after`` on
        the ``slug`` keyword), so the registry has no size cap."""
        by_slug: dict[str, ProjectRecord] = {}
        after: list[Any] | None = None
        while True:
            body: dict[str, Any] = {
                # OpenSearch refuses prefix queries on _id; only project
                # docs carry `slug` (the revision counter doc does not).
                'query': {'exists': {'field': 'slug'}},
                'size': _REFRESH_PAGE_SIZE,
                'sort': [{'slug': 'asc'}],
            }
            if after is not None:
                body['search_after'] = after
            resp = await client.search(index=projects_index(), body=body)
            hits = resp.get('hits', {}).get('hits', [])
            for hit in hits:
                by_slug[hit['_source']['slug']] = doc_to_record(hit['_source'])
            if len(hits) < _REFRESH_PAGE_SIZE:
                break
            after = hits[-1].get('sort') or [hits[-1]['_source']['slug']]
        self._by_slug = by_slug
        self._revision = current_revision

    async def poll_loop(self, *, interval_seconds: float = 1.0) -> None:
        """Background refresh loop; started at API startup and cancelled at
        shutdown. :meth:`ensure_fresh` never raises, so a transient
        OpenSearch hiccup cannot end the loop."""
        while True:
            await self.ensure_fresh()
            await asyncio.sleep(interval_seconds)


_REGISTRY: ProjectRegistry | None = None


def get_project_registry() -> ProjectRegistry:
    """Process-wide registry singleton. Constructed lazily against the
    shared curation OpenSearch client so tests can swap the client
    factory before first use."""
    global _REGISTRY  # noqa: PLW0603 - lazily-built module singleton
    if _REGISTRY is None:

        async def _client_factory() -> Any:
            from src.core.dependencies import get_opensearch

            wrapper = await get_opensearch()
            return getattr(wrapper, 'client', wrapper)

        _REGISTRY = ProjectRegistry(_client_factory)
    return _REGISTRY


def set_project_registry(registry: ProjectRegistry | None) -> None:
    """Test/startup seam."""
    global _REGISTRY  # noqa: PLW0603 - lazily-built module singleton
    _REGISTRY = registry
