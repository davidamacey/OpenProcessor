"""The ``op_projects`` registry: one doc per project plus a revision
counter, and an in-process snapshot refreshed on a cheap poll (one GET of
the counter; a ``_search`` only when it changed).

See ``docs/design/openprocessor_internal/projects_plan.md`` §4.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
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


class ProjectRegistry:
    """In-process snapshot of every project record, kept fresh by
    :meth:`ensure_fresh` (called from the ``bind_path_project`` dependency,
    so a bind is never more than ~1s stale) and by :meth:`poll_loop` (a
    background task started at API startup)."""

    def __init__(self, client_factory: Any) -> None:
        """``client_factory`` is a zero-arg callable (sync or async)
        returning the raw OpenSearch client to query."""
        self._client_factory = client_factory
        self._by_slug: dict[str, ProjectRecord] = {}
        self._revision: int = -1
        self._lock = asyncio.Lock()

    def snapshot(self) -> Mapping[str, ProjectRecord]:
        """The last-refreshed view. Cheap, sync, no I/O -- callers that
        need at-most-1s staleness should call :meth:`ensure_fresh` first."""
        return dict(self._by_slug)

    def get(self, slug: str) -> ProjectRecord | None:
        return self._by_slug.get(slug)

    async def ensure_fresh(self) -> None:
        """One GET of the revision counter; a ``_search`` over every
        project doc only when the counter moved."""
        client = self._client_factory()
        if asyncio.iscoroutine(client):
            client = await client
        try:
            counter_doc = await client.get(index=projects_index(), id=REVISION_DOC_ID)
            current_revision = int((counter_doc.get('_source') or {}).get('revision', 0))
        except Exception:
            current_revision = 0
        if current_revision == self._revision:
            return
        async with self._lock:
            if current_revision == self._revision:
                return
            await self._refresh(client, current_revision)

    async def _refresh(self, client: Any, current_revision: int) -> None:
        resp = await client.search(
            index=projects_index(),
            body={'query': {'prefix': {'_id': 'project:'}}, 'size': 1000},
        )
        hits = resp.get('hits', {}).get('hits', [])
        self._by_slug = {hit['_source']['slug']: doc_to_record(hit['_source']) for hit in hits}
        self._revision = current_revision

    async def poll_loop(self, *, interval_seconds: float = 1.0) -> None:
        """Background refresh loop; started at API startup and cancelled at
        shutdown. Errors are logged and swallowed -- a transient
        OpenSearch hiccup must not crash the process; the next
        request-time ``ensure_fresh`` call still runs."""
        while True:
            with contextlib.suppress(asyncio.CancelledError):
                try:
                    await self.ensure_fresh()
                except Exception:
                    logger.exception('project registry poll failed')
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
