"""TritonControlService reports HTTP failures as a result and lets programming errors surface."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from src.services.triton_control import TritonControlService


class _Client:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *_a: Any) -> None:
        return None

    async def get(self, *_a: Any, **_k: Any) -> Any:
        raise self._exc

    post = get


def _patch(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> TritonControlService:
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **_k: _Client(exc))
    return TritonControlService.__new__(TritonControlService)


@pytest.mark.asyncio
async def test_http_failure_is_a_not_ready_result(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _patch(monkeypatch, httpx.ConnectError('down'))
    svc.http_url = 'http://t'
    assert await svc.server_ready() is False
    assert await svc.get_repository_index() == []


@pytest.mark.asyncio
async def test_a_bug_is_not_reported_as_triton_down(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _patch(monkeypatch, RuntimeError('bug'))
    svc.http_url = 'http://t'
    with pytest.raises(RuntimeError):
        await svc.server_ready()
    with pytest.raises(RuntimeError):
        await svc.get_repository_index()
