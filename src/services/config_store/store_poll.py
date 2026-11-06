"""Background refresh of every active project's config store, and the lifespan hooks
that bootstrap and stop it."""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from src.config.database import DatabaseConfigError
from src.core.logging import get_logger
from src.services.config_store.documents import config_poll_interval_s


logger = get_logger(__name__)

GLOBAL_EPOCH_KEY = '__global__'


async def poll_config_epochs(known: dict[str, int | None]) -> dict[str, int | None]:
    """Read the epoch of every live project and the global one; return those that moved since ``known``.

    The keys are project slugs and ``'__global__'``. Every project's epoch is reported to the
    embedding-space resolver on each read (``note_config_epoch``), so a flip is seen within one poll.
    One indexed read of ``projects`` and one of the counter; nothing touches OpenSearch.
    """
    from sqlalchemy import select

    from src.db.models.projects import Project
    from src.db.repos.config import get_epoch
    from src.db.session import async_session_factory, deployment_tx
    from src.services.embedding_spaces.resolve import note_config_epoch

    async with async_session_factory()() as session, deployment_tx(session, 'poll config epochs'):
        rows = (
            await session.execute(
                select(Project.slug, Project.config_epoch).where(
                    Project.status.in_(('active', 'archived'))
                )
            )
        ).all()
        global_epoch = await get_epoch(session, None)
    current: dict[str, int | None] = {slug: int(epoch) for slug, epoch in rows}
    current[GLOBAL_EPOCH_KEY] = global_epoch
    for slug, epoch in rows:
        note_config_epoch(slug, int(epoch))
    return {key: epoch for key, epoch in current.items() if known.get(key) != epoch}


async def _poll_all_active_projects(interval: float) -> None:
    """Refresh the store of every project whose epoch moved, and the deployment-wide one.

    An activation made through uvicorn worker A is visible in worker B within one tick. Cheap:
    one indexed read of ``projects`` and one of the global counter per tick; a store reloads only
    when its own epoch moved. A store whose refresh fails (it marks itself ``stale``) is retried
    on the next tick, and one project's failure never stops the others.
    """
    from src.services.config_store.global_store import get_global_config_store
    from src.services.config_store.store import config_store_for

    known: dict[str, int | None] = {}
    while True:
        try:
            moved = await poll_config_epochs(known)
        except (SQLAlchemyError, DatabaseConfigError) as exc:
            logger.warning('config_epoch_poll_failed', error=str(exc))
            moved = {}
        for key, epoch in moved.items():
            store = (
                get_global_config_store(mode='live')
                if key == GLOBAL_EPOCH_KEY
                else config_store_for(key, mode='live')
            )
            try:
                snapshot = await store.refresh()
            except Exception as exc:
                logger.warning('config_store_poll_failed', store=key, error=str(exc))
                continue
            if not snapshot.stale:
                known[key] = epoch
        await asyncio.sleep(interval)


async def startup_bootstrap_config_store_safe() -> Any:
    """``src.main``'s lifespan hook: return the background poll task the caller owns
    cancelling at shutdown. Never raises: an unreachable database is logged by the loop and the
    next tick retries, so every worker polls once the database is up."""
    return asyncio.create_task(_poll_all_active_projects(config_poll_interval_s()))


async def shutdown_config_store_poll(task: Any | None) -> None:
    """Cancel the poll task :func:`startup_bootstrap_config_store_safe`
    started, and swallow the resulting ``CancelledError``."""
    import contextlib

    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
