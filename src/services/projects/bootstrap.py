"""Idempotent startup bootstrap of the ``default`` project row.

``default`` is created like any project (indexes ``{OP_PROJECT_INDEX_PREFIX}default__<role>``,
storage prefix ``projects/default``). The bootstrap touches no data index; the project's indexes
are created by the normal per-project index bootstrap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.projects import DEFAULT_SLUG
from src.core.logging import get_logger
from src.db.repos import projects as repo
from src.db.session import async_session_factory, deployment_tx
from src.services.projects.records import new_project_for, record_from_row
from src.services.projects.store import create_with_default_spaces


if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.config.projects import ProjectRecord


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


async def bootstrap_default_project() -> ProjectRecord:
    """Create the ``default`` project if it does not exist yet, or return the stored one unchanged.

    Never overwrites an existing project (a later env change must not remap a live one) and
    touches no data index. The slug claim in the create transaction decides a race between API
    workers: the loser reads the winner's row."""
    async with async_session_factory()() as session:
        try:
            async with deployment_tx(session, 'bootstrap the default project'):
                row = await create_with_default_spaces(
                    session,
                    new_project_for(
                        DEFAULT_SLUG,
                        display_name='Default',
                        description='The project every fresh install starts with.',
                        status='active',
                    ),
                )
        except repo.SlugTakenError:
            async with deployment_tx(session, 'read the default project'):
                existing = await repo.get_by_slug(session, DEFAULT_SLUG)
            if existing is None:
                raise
            return record_from_row(existing)
    logger.info('bootstrapped default project')
    return record_from_row(row)


async def _bootstrap_once() -> Any:
    """Upsert ``default``, read the registry and seed each project's region
    class. Raises while OpenSearch is unreachable or the registry unread."""
    from src.services.projects.registry import get_project_registry

    await bootstrap_default_project()
    registry = get_project_registry()
    await registry.ensure_fresh()
    if not registry.refreshed:
        raise RuntimeError('project registry not readable yet')
    await _seed_region_classes()
    return registry


async def startup_bootstrap_project_registry() -> Any:
    """Everything ``src.main``'s lifespan needs for the projects
    foundation: upsert the ``default`` record, do one registry refresh,
    and return an ``asyncio.Task`` running the background poll loop
    (the caller owns cancelling it at shutdown). Pulled out of
    ``src.main`` to keep that module under the repo's per-file LOC
    ratchet."""
    import asyncio

    registry = await _bootstrap_once()
    return asyncio.create_task(registry.poll_loop())


async def _seed_region_classes() -> None:
    from src.services.curation.region_class import ensure_region_class

    for slug in for_each_project():
        try:
            await ensure_region_class()
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


_BOOTSTRAP_RETRY_INITIAL_S = 1.0
_BOOTSTRAP_RETRY_MAX_S = 30.0


async def startup_bootstrap_project_registry_safe() -> Any | None:
    """``startup_bootstrap_project_registry``, but never raises: when
    OpenSearch is not reachable yet (a fresh install starts everything at
    once), the returned task keeps retrying with backoff and then runs the
    poll loop, so ``default`` still gets created and seeded."""
    import asyncio

    try:
        return await startup_bootstrap_project_registry()
    except Exception as exc:
        logger.warning('project_registry_bootstrap_deferred', error=str(exc))

    async def _retry_then_poll() -> None:
        delay = _BOOTSTRAP_RETRY_INITIAL_S
        while True:
            await asyncio.sleep(delay)
            try:
                registry = await _bootstrap_once()
            except Exception as exc:
                logger.warning('project_registry_bootstrap_retry_failed', error=str(exc))
                delay = min(delay * 2, _BOOTSTRAP_RETRY_MAX_S)
                continue
            logger.info('project_registry_bootstrap_recovered')
            await registry.poll_loop()

    return asyncio.create_task(_retry_then_poll())
