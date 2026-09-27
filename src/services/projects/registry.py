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


async def write_record(
    client: Any,
    record: ProjectRecord,
    *,
    if_seq_no: int | None = None,
    if_primary_term: int | None = None,
) -> None:
    """Write ``record`` (create or OCC-guarded overwrite) and bump the
    registry revision. Raises the client's version-conflict exception
    (409-shaped) when ``if_seq_no``/``if_primary_term`` are stale --
    callers translate that into ``revision_conflict``."""
    from src.services.projects.bootstrap import bump_revision

    kwargs: dict[str, Any] = {}
    if if_seq_no is not None:
        kwargs['if_seq_no'] = if_seq_no
    if if_primary_term is not None:
        kwargs['if_primary_term'] = if_primary_term
    await client.index(
        index=projects_index(),
        id=_project_doc_id(record.slug),
        body=record_to_doc(record),
        **kwargs,
    )
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

    def snapshot(self) -> Mapping[str, ProjectRecord]:
        """The last-refreshed view. Cheap, sync, no I/O -- callers that
        need at-most-1s staleness should call :meth:`ensure_fresh` first."""
        return dict(self._by_slug)

    def get(self, slug: str) -> ProjectRecord | None:
        return self._by_slug.get(slug)

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

    async def _refresh(self, client: Any, current_revision: int) -> None:
        """Read every project doc, a page at a time (``search_after`` on
        the ``slug`` keyword), so the registry has no size cap."""
        by_slug: dict[str, ProjectRecord] = {}
        after: list[Any] | None = None
        while True:
            body: dict[str, Any] = {
                'query': {'prefix': {'_id': 'project:'}},
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
