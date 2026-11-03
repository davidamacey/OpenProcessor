"""A by-name VLM lookup that misses forces one registry refresh (#442).

``ensure_fresh`` trusts a snapshot younger than a second, so an endpoint created
a moment earlier (through another API worker) read as unknown to ``activate``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.services import config_store
from src.services.labeling import vlm_endpoints


class _Store:
    def __init__(self) -> None:
        self.known = False
        self.forced = 0

    async def ensure_fresh(self, _client: Any) -> None:
        return None  # the snapshot is "fresh": no reload

    async def refresh(self, _client: Any, *, force: bool = False) -> None:
        if force:
            self.forced += 1
            self.known = True


def test_miss_forces_refresh_then_resolves(monkeypatch) -> None:
    store = _Store()
    sentinel = object()
    monkeypatch.setattr(config_store, 'get_global_config_store', lambda: store)
    monkeypatch.setattr(
        vlm_endpoints, 'get_vlm_endpoint', lambda _n: sentinel if store.known else None
    )
    got = asyncio.run(vlm_endpoints.resolve_vlm_source(None, 'new_endpoint'))
    assert got is sentinel
    assert store.forced == 1


def test_still_unknown_after_refresh_is_none(monkeypatch) -> None:
    store = _Store()
    monkeypatch.setattr(config_store, 'get_global_config_store', lambda: store)
    monkeypatch.setattr(vlm_endpoints, 'get_vlm_endpoint', lambda _n: None)
    assert asyncio.run(vlm_endpoints.resolve_vlm_source(None, 'nope')) is None
    assert store.forced == 1
