"""Sync Triton callers route through the process router (spread, model-aware, fail closed)."""

from __future__ import annotations

import pytest

from src.clients.triton_router import NoServerError, TritonRouter
from src.services import sync_triton


class _Pool:
    def __init__(self, router: TritonRouter) -> None:
        self.router = router


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch):
    from src.app.resources import AppResources

    monkeypatch.setattr(sync_triton, 'TritonClient', lambda url: ('client', url))
    sync_triton._triton_client.cache_clear()
    yield
    monkeypatch.setattr(AppResources, 'async_triton_pool', None)
    sync_triton._triton_client.cache_clear()


def _bind(monkeypatch: pytest.MonkeyPatch, router: TritonRouter) -> None:
    from src.app.resources import AppResources

    monkeypatch.setattr(AppResources, 'async_triton_pool', _Pool(router))


def _two(ready: dict[str, list[str]]) -> TritonRouter:
    router = TritonRouter({'a': 'a:8001', 'b': 'b:8001'})
    router.update_ready(ready)
    return router


def test_no_pool_is_the_single_default_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sync_triton, 'get_settings', lambda: type('S', (), {'triton_url': 'd:8001'})
    )
    got: object = sync_triton.routed_triton_client('any-model')
    assert got == ('client', 'd:8001')


def test_callers_spread_across_two_ready_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _two({'a': ['m'], 'b': ['m']})
    _bind(monkeypatch, router)
    first = sync_triton.routed_url('m')
    with router.route('m') as busy:  # the event loop's callers load one server
        assert sync_triton.routed_url('m') != busy.url
    assert first in {'a:8001', 'b:8001'}


def test_a_server_without_the_model_gets_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _bind(monkeypatch, _two({'a': ['m'], 'b': ['other']}))
    assert {sync_triton.routed_url('m') for _ in range(5)} == {'a:8001'}


def test_every_model_of_a_call_must_be_on_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    _bind(monkeypatch, _two({'a': ['yolo'], 'b': ['yolo', 'clip']}))
    assert sync_triton.routed_url('yolo', 'clip') == 'b:8001'


def test_no_ready_server_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _bind(monkeypatch, _two({'a': [], 'b': ['other']}))
    with pytest.raises(NoServerError):
        sync_triton.routed_triton_client('m')


def test_one_wrapper_per_server_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _bind(monkeypatch, _two({'a': ['m'], 'b': []}))
    assert sync_triton.routed_triton_client('m') is sync_triton.routed_triton_client('m')
