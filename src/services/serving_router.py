"""Build the per-process Triton router from the active serving profile and its reports.

The servers come from the active ``serving_profile`` (``src/services/config_store/research_kinds.py``);
without one the router is the single default server, exactly the behaviour before routing. Which
models each server serves comes from ``serving_reports``, which ``op.periodic.serving_inventory``
keeps current; only rows in state ``ready`` count, so a server the inventory could not reach (all
rows ``unavailable``) gets no call.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import text

from src.clients.triton_router import TritonRouter
from src.db.session import async_session_factory, deployment_tx
from src.services.config_store.research_kinds import active_serving_profile


_READY = text("SELECT server, model FROM serving_reports WHERE state = 'ready'")


async def ready_models() -> dict[str, set[str]]:
    """Per server, the models ``serving_reports`` says are ready."""
    ready: dict[str, set[str]] = defaultdict(set)
    async with async_session_factory()() as session, deployment_tx(session, 'serving'):
        for server, model in (await session.execute(_READY)).all():
            ready[server].add(model)
    return ready


async def build_router(default_url: str) -> TritonRouter:
    """The router for the active profile, or the single default server when none is active."""
    active = await active_serving_profile()
    if active is None:
        return TritonRouter.single(default_url)
    router = TritonRouter({s['name']: s['grpc_url'] for s in active[0].body['servers']})
    await refresh_ready(router)
    return router


async def refresh_ready(router: TritonRouter) -> None:
    router.update_ready(await ready_models())
