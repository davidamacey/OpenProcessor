"""Build the per-process Triton router from the active serving profile and its reports.

The servers come from the active ``serving_profile`` (``src/services/config_store/research_kinds.py``);
without one the router is the single default server, exactly the behaviour before routing. Which
models each server serves comes from ``serving_reports``, which ``op.periodic.serving_inventory``
keeps current (the API refreshes it on its config poll tick, a worker on each job); only rows in state ``ready`` count, so a server the inventory could not reach (all
rows ``unavailable``) gets no call.

A serving-profile change is picked up on the same tick (``sync_router``): the router records the
setting revision it was built from, and when the active revision differs the old router refuses
calls, a new one is built and the pool adopts it (channels opened to the new servers, the dropped
ones closed). A rebuild that fails leaves the old router refusing, and the next tick retries.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.clients.triton_pool_core import AsyncTritonPool
from src.clients.triton_router import TritonRouter
from src.config import get_settings
from src.db.session import async_session_factory, deployment_tx
from src.services.config_store.research_kinds import active_serving_profile


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


_READY = text("SELECT server, model FROM serving_reports WHERE state = 'ready'")


async def ready_models() -> dict[str, set[str]]:
    """Per server, the models ``serving_reports`` says are ready."""
    ready: dict[str, set[str]] = defaultdict(set)
    async with async_session_factory()() as session, deployment_tx(session, 'serving'):
        for server, model in (await session.execute(_READY)).all():
            ready[server].add(model)
    return ready


async def _router_for(default_url: str, active: tuple[Any, int] | None) -> TritonRouter:
    if active is None:
        return TritonRouter.single(default_url)
    doc, revision = active
    router = TritonRouter(
        {s['name']: s['grpc_url'] for s in doc.body['servers']}, profile_revision=revision
    )
    await refresh_ready(router)
    return router


async def build_router(default_url: str) -> TritonRouter:
    """The router for the active profile, or the single default server when none is active."""
    return await _router_for(default_url, await active_serving_profile())


async def open_routed_pool(
    *, pool_size: int, max_concurrent: int, verbose: bool = False, url: str | None = None
) -> AsyncTritonPool:
    """An initialized pool whose router follows the active serving profile; ``url`` is the
    default server when no profile is active (the deployment's ``TRITON_URL`` unless given)."""
    url = url or get_settings().triton_url
    pool = AsyncTritonPool(
        url=url,
        pool_size=pool_size,
        max_concurrent=max_concurrent,
        verbose=verbose,
        router=await build_router(url),
    )
    await pool.initialize()
    return pool


async def refresh_ready(router: TritonRouter) -> None:
    """Re-read the ready rows; the single default server follows no reports and is left alone."""
    if router.follows_reports:
        router.update_ready(await ready_models())


async def sync_router(pool: AsyncTritonPool) -> None:
    """Bring the pool's router up to date: rebuild it when the active profile revision moved
    (refusing calls until the new one is in place), else re-read the ready rows."""
    active = await active_serving_profile()
    revision = None if active is None else active[1]
    if revision == pool.router.profile_revision and not pool.router.rebuilding:
        await refresh_ready(pool.router)
        return
    pool.router.hold_for_rebuild()
    await pool.adopt_router(await _router_for(pool.url, active))


def refresh_tick(pool: AsyncTritonPool) -> Callable[[], Awaitable[None]]:
    """The periodic router sync to ride an existing poll loop."""

    async def tick() -> None:
        await sync_router(pool)

    return tick
