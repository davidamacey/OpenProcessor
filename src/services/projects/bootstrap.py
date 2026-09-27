"""Idempotent startup bootstrap of the ``default`` project record.

``default`` is created like any project (``new_project_record``: indexes
``{OP_PROJECT_INDEX_PREFIX}default__<role>``, dirs under the projects data
and state roots). The bootstrap touches no data index; the project's
indexes are created by the normal per-project index bootstrap.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.curation import base_curation_config
from src.config.projects import DEFAULT_SLUG, ProjectRecord, new_project_record
from src.core.logging import get_logger
from src.services.projects.registry import REVISION_DOC_ID, projects_index, record_to_doc


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = get_logger(__name__)


def for_each_project(statuses: tuple[str, ...] = ('active',)) -> Iterator[str]:
    """Bind each project in ``statuses`` in turn (sorted by slug) and
    yield its slug; the binding holds for the body of the caller's loop.

    For startup steps that act on project data (index bootstrap, kNN
    warmup, orphaned-job reconcile). The lifespan itself stays unbound, so
    a global loop never silently acts on ``default``."""
    from src.config.project_context import bind_project
    from src.services.projects.registry import get_project_registry

    for slug, record in sorted(get_project_registry().snapshot().items()):
        if record.status not in statuses:
            continue
        with bind_project(record):
            yield slug


async def bootstrap_default_project(client: Any) -> ProjectRecord:
    """Create the ``default`` project doc if it does not exist yet, or
    return the existing one unchanged. Never overwrites an existing
    record (a later env change must not remap a live project), and never
    touches any data index."""
    from src.services.projects.guard import bind_registry_admin
    from src.services.projects.registry import doc_to_record

    doc_id = f'project:{DEFAULT_SLUG}'
    with bind_registry_admin():
        await ensure_projects_index(client)
        # Only a definite "not found" means first boot. Any other failure
        # (a 503 right after the cluster starts, a timeout, a malformed
        # doc) propagates, so startup retries later instead of resetting
        # an existing record (e.g. an archived ``default``).
        existing = await _get_or_none(client, doc_id)
        if existing is not None:
            return doc_to_record(existing)

        record = new_project_record(
            DEFAULT_SLUG,
            base_curation_config(),
            display_name='Default',
            description='The project every fresh install starts with.',
            now=datetime.now(UTC).isoformat(),
        )
        try:
            await client.index(
                index=projects_index(), id=doc_id, body=record_to_doc(record), op_type='create'
            )
        except Exception as exc:
            if not _is_conflict(exc):
                raise
            # Another worker created it first: its record wins.
            winner = await _get_or_none(client, doc_id)
            if winner is None:
                raise
            return doc_to_record(winner)
        await bump_revision(client)
    logger.info('bootstrapped default project record')
    return record


def _is_not_found(exc: BaseException) -> bool:
    return getattr(exc, 'status_code', None) == 404 or 'NotFound' in type(exc).__name__


def _is_conflict(exc: BaseException) -> bool:
    return getattr(exc, 'status_code', None) == 409 or 'Conflict' in type(exc).__name__


async def _get_or_none(client: Any, doc_id: str) -> dict[str, Any] | None:
    """The stored doc's ``_source``, or ``None`` only when OpenSearch says
    the doc does not exist. Every other failure is raised."""
    try:
        existing = await client.get(index=projects_index(), id=doc_id)
    except Exception as exc:
        if _is_not_found(exc):
            return None
        raise
    if not existing.get('found', True):
        return None
    return existing['_source']


# ``dynamic: false``: the registry stores whole records, but only these
# fields are ever queried; nothing a record carries can change the mapping.
PROJECTS_INDEX_BODY: dict[str, Any] = {
    'settings': {'index': {'number_of_shards': 1, 'number_of_replicas': 0}},
    'mappings': {
        'dynamic': False,
        'properties': {
            'slug': {'type': 'keyword'},
            'status': {'type': 'keyword'},
            'revision': {'type': 'long'},
            'created_at': {'type': 'date'},
            'updated_at': {'type': 'date'},
        },
    },
}


async def ensure_projects_index(client: Any) -> None:
    """Create ``op_projects`` with its explicit mapping if it does not
    exist, so the first registry write never auto-creates it with a
    dynamic mapping. Call inside :func:`bind_registry_admin`."""
    if await client.indices.exists(index=projects_index()):
        return
    await client.indices.create(index=projects_index(), body=PROJECTS_INDEX_BODY)


_BUMP_RETRIES = 5


async def bump_revision(client: Any) -> int:
    """Increment the ``meta:projects_revision`` counter with optimistic
    concurrency (``if_seq_no``/``if_primary_term``; ``op_type=create`` on
    the first write), retrying on a conflict. Returns the new revision.
    Call inside :func:`bind_registry_admin`."""
    for _attempt in range(_BUMP_RETRIES):
        try:
            current = await client.get(index=projects_index(), id=REVISION_DOC_ID)
        except Exception as exc:
            if not _is_not_found(exc):
                raise
            current = {'found': False}
        try:
            if current.get('found', True) and '_source' in current:
                revision = int(current['_source'].get('revision', 0)) + 1
                await client.index(
                    index=projects_index(),
                    id=REVISION_DOC_ID,
                    body={'revision': revision},
                    if_seq_no=current.get('_seq_no'),
                    if_primary_term=current.get('_primary_term'),
                )
            else:
                revision = 1
                await client.index(
                    index=projects_index(),
                    id=REVISION_DOC_ID,
                    body={'revision': revision},
                    op_type='create',
                )
        except Exception as exc:
            if _is_conflict(exc):
                continue
            raise
        return revision
    raise RuntimeError('project registry revision kept conflicting; giving up')


async def startup_bootstrap_project_registry() -> Any:
    """Everything ``src.main``'s lifespan needs for the projects
    foundation: upsert the ``default`` record, do one registry refresh,
    and return an ``asyncio.Task`` running the background poll loop
    (the caller owns cancelling it at shutdown). Pulled out of
    ``src.main`` to keep that module under the repo's per-file LOC
    ratchet."""
    import asyncio

    from src.services.projects.guard import make_curation_opensearch
    from src.services.projects.registry import get_project_registry

    client = await make_curation_opensearch()
    await bootstrap_default_project(client)
    registry = get_project_registry()
    await registry.ensure_fresh()
    _seed_region_classes()
    return asyncio.create_task(registry.poll_loop())


def _seed_region_classes() -> None:
    from src.services.curation.region_class import ensure_region_class

    for slug in for_each_project():
        try:
            ensure_region_class()
        except Exception as exc:
            logger.warning('region_class_seed_failed', project=slug, error=str(exc))


async def shutdown_project_registry(task: Any | None) -> None:
    """Cancel the poll task ``startup_bootstrap_project_registry``
    started, and swallow the resulting ``CancelledError``."""
    import asyncio
    import contextlib

    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def startup_bootstrap_project_registry_safe() -> Any | None:
    """``startup_bootstrap_project_registry``, but never raises -- a
    startup-time OpenSearch hiccup here must not block the rest of the
    app from starting; the next request-time ``ensure_fresh()`` call
    still runs."""
    try:
        return await startup_bootstrap_project_registry()
    except Exception as exc:
        logger.warning('project_registry_bootstrap_skipped', error=str(exc))
        return None
