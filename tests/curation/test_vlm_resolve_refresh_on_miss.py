"""A by-name VLM lookup that misses forces one registry refresh (#442).

``ensure_fresh`` trusts a snapshot younger than a second, so an endpoint created
a moment earlier (through another API worker) read as unknown to ``activate``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.services import config_store
from src.services.labeling import vlm_endpoints


class _Store:
    def __init__(self) -> None:
        self.known = False
        self.forced = 0

    async def ensure_fresh(self) -> None:
        return None  # the snapshot is "fresh": no reload

    async def refresh_on_miss(self) -> None:
        self.forced += 1
        self.known = True


def test_miss_forces_refresh_then_resolves(monkeypatch) -> None:
    store = _Store()
    sentinel = object()
    monkeypatch.setattr(config_store, 'get_global_config_store', lambda: store)
    monkeypatch.setattr(
        vlm_endpoints, 'get_vlm_endpoint', lambda _n: sentinel if store.known else None
    )
    got = asyncio.run(vlm_endpoints.resolve_vlm_source('new_endpoint'))
    assert got is sentinel
    assert store.forced == 1


def test_still_unknown_after_refresh_is_none(monkeypatch) -> None:
    store = _Store()
    monkeypatch.setattr(config_store, 'get_global_config_store', lambda: store)
    monkeypatch.setattr(vlm_endpoints, 'get_vlm_endpoint', lambda _n: None)
    assert asyncio.run(vlm_endpoints.resolve_vlm_source('nope')) is None
    assert store.forced == 1


def test_unreadable_registry_is_unavailable_not_unknown(monkeypatch) -> None:
    from src.services.config_store.store import ConfigStoreUnavailableError

    class _Down(_Store):
        async def refresh_on_miss(self) -> None:
            raise ConfigStoreUnavailableError('down')

    monkeypatch.setattr(config_store, 'get_global_config_store', _Down)
    monkeypatch.setattr(vlm_endpoints, 'get_vlm_endpoint', lambda _n: None)
    with pytest.raises(vlm_endpoints.VlmRegistryUnavailableError):
        asyncio.run(vlm_endpoints.resolve_vlm_source('nope'))


def test_run_route_answers_503_when_registry_unreadable(monkeypatch) -> None:
    from src.core.exceptions import RequestRejectedError
    from src.services.curation.autolabel import run_vlm

    async def _boom(*_a: Any, **_k: Any) -> None:
        raise vlm_endpoints.VlmRegistryUnavailableError('registry down')

    async def _noop() -> None:
        return None

    monkeypatch.setattr(run_vlm, 'refresh_vlm_state', _noop)
    monkeypatch.setattr(run_vlm, 'resolve_vlm_source', _boom)
    with pytest.raises(RequestRejectedError) as err:
        asyncio.run(run_vlm.resolve_run_vlm('ghost', pack=None))
    assert err.value.status_code == 503
