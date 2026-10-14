"""Background refresh of every active project's config store, and the lifespan hooks
that bootstrap and stop it."""

from __future__ import annotations

import asyncio
from typing import Any

from opensearchpy.exceptions import OpenSearchException

from src.core.logging import get_logger
from src.services.config_store.index import config_poll_interval_s
from src.services.config_store.store import get_config_store


logger = get_logger(__name__)

#: Default interval (seconds) for ``poll_loop`` -- fast enough that a
#: pack/profile activated in one uvicorn worker is visible in every
#: other within one tick, cheap enough (one ``GET`` per project per
#: tick when nothing changed) to run continuously.


async def _poll_all_active_projects(client: Any, interval: float) -> None:
    """M4: refresh every ACTIVE project's own store on each tick, not
    just the one bound at lifespan startup -- otherwise an activation
    made through uvicorn worker A stays invisible in worker B until
    some other route happens to call that project's ``ensure_fresh()``
    (only the two settings routes did). Cheap: one ``GET`` per active
    project per tick when nothing changed (:meth:`ConfigStore.refresh`
    short-circuits on an unmoved revision counter). A single project's
    refresh failure is logged and never stops the loop or the other
    projects' refreshes -- :meth:`ConfigStore.refresh` already swallows
    its own errors (marks ``stale``), so this only guards the registry
    read + project iteration itself.
    """
    import asyncio

    from src.config.project_context import bind_project
    from src.services.config_store.global_store import get_global_config_store
    from src.services.projects.registry import ProjectRegistry

    registry = ProjectRegistry(lambda: client)
    while True:
        try:
            # the deployment-wide VLM registry, refreshed unbound (its
            # index belongs to no project) so a probe or an endpoint edit
            # made through another uvicorn worker shows up here within a tick.
            await get_global_config_store(mode='live').refresh(client)
        except OpenSearchException as exc:
            logger.warning('config_store_poll_global_failed', error=str(exc))
        try:
            await registry.ensure_fresh()
            for record in registry.active_projects():
                try:
                    with bind_project(record, read_only=True):
                        store = get_config_store(mode='live')
                        await store.refresh(client)
                except Exception as exc:
                    logger.warning(
                        'config_store_poll_project_failed', project=record.slug, error=str(exc)
                    )
        except Exception as exc:
            logger.warning('config_store_poll_registry_failed', error=str(exc))
        await asyncio.sleep(interval)


_CONFIG_STORE_BOOTSTRAP_RETRY_INITIAL_S = 1.0
_CONFIG_STORE_BOOTSTRAP_RETRY_MAX_S = 30.0


async def _bootstrap_config_store_once() -> tuple[Any, float]:
    """Ensure ``op_global_configs`` exists (M3) and resolve the poll
    interval. Raises while OpenSearch is unreachable, or while a lost
    index-create race (``resource_already_exists_exception`` from
    ``indices.create``, another worker having just won it) hasn't yet
    resolved into ``indices.exists`` seeing the winner's index -- the
    caller retries until it doesn't."""
    from src.services.config_store.global_store import ensure_global_configs_index
    from src.services.projects.guard import make_curation_opensearch

    client = await make_curation_opensearch()
    # M3: needs no project bound (this index belongs to none) -- runs
    # first, so a later failure below never skips it.
    await ensure_global_configs_index(client)
    interval = config_poll_interval_s()
    return client, interval


async def startup_bootstrap_config_store_safe() -> Any:
    """``src.main``'s lifespan hook: create ``op_global_configs`` (M3) if
    it does not exist yet, then return a background poll task -- fanned
    out over every ACTIVE project (M4), not just the one bound at
    lifespan startup -- that the caller owns cancelling at shutdown.

    Never raises, and always returns a real, running task. If OpenSearch
    is unreachable, or a lost index-create race surfaces as
    ``resource_already_exists_exception`` before ``indices.exists`` sees
    the winner's index, the task keeps retrying
    :func:`_bootstrap_config_store_once` with backoff instead of giving
    up, then enters :func:`_poll_all_active_projects` once it succeeds --
    so that first tick genuinely does eventually run, on every worker,
    every time.

    This used to also call
    ``get_config_store(mode='live')`` + ``store.refresh(client)`` to warm
    "the bound project's store" -- but the lifespan runs unbound by
    design (``src.main``'s startup binds each project in turn only for
    the steps that need it), so that call always raised
    ``ProjectNotBound``. The broad ``except`` below then swallowed it and
    returned ``None`` on every real deployment, so the poll task never
    started at all. There is no bound project to warm here;
    ``_poll_all_active_projects`` binds and refreshes every active
    project on its own first tick.

    An earlier fix still ran
    ``ensure_global_configs_index`` inline before ``create_task``, inside
    a broad ``except`` that returned ``None`` on any failure there --
    including a plain unreachable-OpenSearch startup or a lost
    index-create race, both realistic at cold-stack-start (production
    ``api`` has no ``opensearch: service_healthy`` gate and runs 32
    workers). Nothing then ever retried, so that worker had no poll task
    for its entire lifetime (M4 inert), contrary to what this docstring
    used to claim. Fixed by mirroring
    ``src.services.projects.bootstrap.startup_bootstrap_project_registry_safe``'s
    retry-then-poll shape: try once inline, and if that fails, hand off
    to a task that keeps retrying with backoff until it succeeds, then
    runs the poll loop forever.
    """
    try:
        client, interval = await _bootstrap_config_store_once()
        return asyncio.create_task(_poll_all_active_projects(client, interval))
    except Exception as exc:
        logger.warning('config_store_bootstrap_deferred', error=str(exc))

    async def _retry_then_poll() -> None:
        delay = _CONFIG_STORE_BOOTSTRAP_RETRY_INITIAL_S
        while True:
            await asyncio.sleep(delay)
            try:
                client, interval = await _bootstrap_config_store_once()
            except Exception as exc:
                logger.warning('config_store_bootstrap_retry_failed', error=str(exc))
                delay = min(delay * 2, _CONFIG_STORE_BOOTSTRAP_RETRY_MAX_S)
                continue
            logger.info('config_store_bootstrap_recovered')
            await _poll_all_active_projects(client, interval)

    return asyncio.create_task(_retry_then_poll())


async def shutdown_config_store_poll(task: Any | None) -> None:
    """Cancel the poll task :func:`startup_bootstrap_config_store_safe`
    started, and swallow the resulting ``CancelledError``."""
    import asyncio
    import contextlib

    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
