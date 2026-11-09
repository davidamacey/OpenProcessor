"""The router is built from the active serving profile and the ready rows (fakes, no database)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.clients.triton_router import NoServerError
from src.services import serving_router


def _profile(*servers: tuple[str, str]) -> tuple[Any, int]:
    body = {'servers': [{'name': n, 'grpc_url': u} for n, u in servers]}
    return SimpleNamespace(body=body), 1


@pytest.mark.asyncio
async def test_no_active_profile_is_the_single_default_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def none() -> None:
        return None

    monkeypatch.setattr(serving_router, 'active_serving_profile', none)
    router = await serving_router.build_router('triton-server:8001')
    with router.route('any') as route:
        assert route.url == 'triton-server:8001'


@pytest.mark.asyncio
async def test_servers_come_from_the_profile_and_only_ready_ones_are_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def active() -> tuple[Any, int]:
        return _profile(('gpu0', 'g0:8001'), ('gpu1', 'g1:8001'))

    async def ready() -> dict[str, set[str]]:
        return {'gpu1': {'yolo'}}

    monkeypatch.setattr(serving_router, 'active_serving_profile', active)
    monkeypatch.setattr(serving_router, 'ready_models', ready)
    router = await serving_router.build_router('unused:8001')
    with router.route('yolo') as route:
        assert (route.server, route.url) == ('gpu1', 'g1:8001')
    with pytest.raises(NoServerError), router.route('other'):
        pass


@pytest.mark.asyncio
async def test_the_poll_loop_runs_the_router_refresh_each_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import contextlib

    from src.clients.triton_router import TritonRouter
    from src.services.config_store.store_poll import _poll_all_active_projects

    async def no_moves(known: dict[str, int | None]) -> dict[str, int | None]:
        return {}

    reads = 0

    async def ready() -> dict[str, set[str]]:
        nonlocal reads
        reads += 1
        return {'a': {'yolo'}}

    monkeypatch.setattr('src.services.config_store.store_poll.poll_config_epochs', no_moves)
    monkeypatch.setattr(serving_router, 'ready_models', ready)
    router = TritonRouter({'a': 'a:8001'})
    tick = serving_router.refresh_tick(router)
    assert tick is not None
    task = asyncio.get_event_loop().create_task(_poll_all_active_projects(0.01, tick))
    await asyncio.sleep(0.1)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert reads >= 2
    with router.route('yolo') as route:
        assert route.server == 'a'


def test_the_single_default_server_has_no_refresh() -> None:
    from src.clients.triton_router import TritonRouter

    assert serving_router.refresh_tick(TritonRouter.single('t:8001')) is None
