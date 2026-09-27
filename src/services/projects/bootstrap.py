"""Idempotent startup bootstrap of the ``default`` project record.

This is the *only* "migration" P1 performs, and it touches no data
index -- it just upserts the registry doc so ``default`` shows up in
``GET /projects`` and ``bind_default_project`` has a record to bind.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.config.curation import base_curation_config
from src.config.projects import DEFAULT_SLUG, ProjectRecord, resources_for_default
from src.core.logging import get_logger
from src.services.projects.registry import bump_revision, projects_index, record_to_doc


logger = get_logger(__name__)


def bind_default_for_lifespan() -> None:
    """Bind ``default`` for the API lifespan task.

    Startup work and the background loops the lifespan starts (index
    bootstrap, kNN warmup, orphaned-job reconcile, GPU arbiter) act on the
    ``default`` project; every ``create_task`` in the lifespan inherits
    this binding. Requests never see it: each runs in its own task and
    binds through its route dependency. Iterating every project in these
    loops is P2 (projects_plan.md §5)."""
    from src.config.project_context import set_bound_project
    from src.services.projects.registry import default_project_record

    set_bound_project(default_project_record())


async def bootstrap_default_project(client: Any) -> ProjectRecord:
    """Create the ``default`` project doc if it does not exist yet, or
    return the existing one unchanged. Never overwrites an existing
    record (a later env change must not remap a live project), and never
    issues a reindex/update_by_query against any data index."""
    doc_id = f'project:{DEFAULT_SLUG}'
    try:
        existing = await client.get(index=projects_index(), id=doc_id)
        if existing.get('found', True):
            from src.services.projects.registry import doc_to_record

            return doc_to_record(existing['_source'])
    except Exception:  # nosec B110 - the client raises on a missing doc; that's first-boot, not an error
        logger.debug('no existing default project doc; bootstrapping one')

    now = datetime.now(UTC).isoformat()
    record = ProjectRecord(
        slug=DEFAULT_SLUG,
        display_name='Default',
        description='The original, unscoped dataset workspace.',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_default(base_curation_config()),
    )
    await client.index(index=projects_index(), id=doc_id, body=record_to_doc(record))
    await bump_revision(client)
    logger.info('bootstrapped default project record')
    return record


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
    return asyncio.create_task(registry.poll_loop())


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
