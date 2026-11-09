"""The router is built from the active serving profile and the ready rows (fakes, no database)."""

from __future__ import annotations

import asyncio
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
    import contextlib

    from src.clients.triton_pool_core import AsyncTritonPool
    from src.clients.triton_router import TritonRouter
    from src.services.config_store.store_poll import _poll_all_active_projects

    async def no_moves(known: dict[str, int | None]) -> dict[str, int | None]:
        return {}

    reads = 0

    async def ready() -> dict[str, set[str]]:
        nonlocal reads
        reads += 1
        return {'a': {'yolo'}}

    async def active() -> tuple[Any, int]:
        return _profile(('a', 'a:8001'))

    monkeypatch.setattr('src.services.config_store.store_poll.poll_config_epochs', no_moves)
    monkeypatch.setattr(serving_router, 'ready_models', ready)
    monkeypatch.setattr(serving_router, 'active_serving_profile', active)
    router = TritonRouter({'a': 'a:8001'}, profile_revision=1)
    pool = AsyncTritonPool(url='a:8001', pool_size=1, router=router)
    tick = serving_router.refresh_tick(pool)
    task = asyncio.get_event_loop().create_task(_poll_all_active_projects(0.01, tick))
    await asyncio.sleep(0.1)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert reads >= 2
    assert pool.router is router
    with router.route('yolo') as route:
        assert route.server == 'a'


class _Profiles:
    """The active serving profile as the tests move it: (servers, revision) or None."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now: tuple[Any, int] | None = None
        self.ready: dict[str, set[str]] = {}
        self.ready_gate: asyncio.Event | None = None
        monkeypatch.setattr(serving_router, 'active_serving_profile', self._active)
        monkeypatch.setattr(serving_router, 'ready_models', self._ready)

    async def _active(self) -> tuple[Any, int] | None:
        return self.now

    async def _ready(self) -> dict[str, set[str]]:
        if self.ready_gate is not None:
            await self.ready_gate.wait()
        return self.ready

    def activate(self, revision: int, *servers: tuple[str, str]) -> None:
        self.now = (_profile(*servers)[0], revision)
        self.ready = {name: {'yolo'} for name, _ in servers}


async def _pool_for(profiles: _Profiles) -> Any:
    from src.clients.triton_pool_core import AsyncTritonPool

    router = await serving_router.build_router('a:8001')
    pool = AsyncTritonPool(url='a:8001', pool_size=1, router=router)
    await pool.initialize()
    return pool


@pytest.mark.asyncio
async def test_a_new_profile_revision_rebuilds_the_router_and_the_pool_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = _Profiles(monkeypatch)
    profiles.activate(1, ('a', 'a:8001'), ('b', 'b:8001'))
    pool = await _pool_for(profiles)
    first = pool.router
    tick = serving_router.refresh_tick(pool)

    await tick()
    assert pool.router is first  # same revision: only the ready rows are re-read

    profiles.activate(2, ('b', 'b:8001'), ('c', 'c:8001'))
    await tick()

    assert pool.router is not first
    assert pool.router.profile_revision == 2
    assert set(pool._servers) == {'b:8001', 'c:8001'}
    with pool.router.route('yolo') as route:
        assert route.url in {'b:8001', 'c:8001'}
    await pool.close()


@pytest.mark.asyncio
async def test_an_activated_then_deactivated_profile_moves_the_router_each_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = _Profiles(monkeypatch)
    pool = await _pool_for(profiles)
    assert not pool.router.follows_reports
    tick = serving_router.refresh_tick(pool)

    profiles.activate(1, ('x', 'x:8001'))
    await tick()
    assert pool.router.follows_reports
    assert set(pool._servers) == {'x:8001'}

    profiles.now = None
    await tick()
    assert not pool.router.follows_reports
    assert set(pool._servers) == {'a:8001'}
    await pool.close()


@pytest.mark.asyncio
async def test_calls_are_refused_while_the_router_is_being_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.clients.triton_router import NoServerError

    profiles = _Profiles(monkeypatch)
    profiles.activate(1, ('a', 'a:8001'))
    pool = await _pool_for(profiles)
    profiles.activate(2, ('b', 'b:8001'))
    profiles.ready_gate = asyncio.Event()
    rebuilding = asyncio.create_task(serving_router.sync_router(pool))
    await asyncio.sleep(0.05)
    with pytest.raises(NoServerError, match='rebuil'), pool.router.route('yolo'):
        pass
    profiles.ready_gate.set()
    await rebuilding
    with pool.router.route('yolo') as route:
        assert route.url == 'b:8001'
    await pool.close()


@pytest.mark.asyncio
async def test_a_failed_rebuild_keeps_refusing_and_the_next_tick_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.exc import OperationalError

    from src.clients.triton_router import NoServerError

    profiles = _Profiles(monkeypatch)
    profiles.activate(1, ('a', 'a:8001'))
    pool = await _pool_for(profiles)
    tick = serving_router.refresh_tick(pool)

    profiles.activate(2, ('b', 'b:8001'))
    good_ready = profiles._ready

    async def broken() -> dict[str, set[str]]:
        raise OperationalError('SELECT', {}, Exception('database down'))

    monkeypatch.setattr(serving_router, 'ready_models', broken)
    with pytest.raises(OperationalError):
        await tick()
    with pytest.raises(NoServerError, match='rebuil'), pool.router.route('yolo'):
        pass

    monkeypatch.setattr(serving_router, 'ready_models', good_ready)
    await tick()
    with pool.router.route('yolo') as route:
        assert route.url == 'b:8001'
    await pool.close()


@pytest.mark.asyncio
async def test_a_profile_that_returns_to_the_held_revision_still_replaces_the_held_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.exc import OperationalError

    profiles = _Profiles(monkeypatch)
    profiles.activate(1, ('a', 'a:8001'))
    pool = await _pool_for(profiles)
    tick = serving_router.refresh_tick(pool)
    good_ready = profiles._ready

    profiles.activate(2, ('b', 'b:8001'))

    async def broken() -> dict[str, set[str]]:
        raise OperationalError('SELECT', {}, Exception('database down'))

    monkeypatch.setattr(serving_router, 'ready_models', broken)
    with pytest.raises(OperationalError):
        await tick()

    monkeypatch.setattr(serving_router, 'ready_models', good_ready)
    profiles.activate(1, ('a', 'a:8001'))
    await tick()
    assert not pool.router.rebuilding
    with pool.router.route('yolo') as route:
        assert route.url == 'a:8001'
    await pool.close()
