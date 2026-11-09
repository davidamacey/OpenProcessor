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
