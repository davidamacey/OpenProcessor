"""TritonControlService reports an HTTP failure as a not-ready result or a typed UpstreamError
(never an empty repository) and lets programming errors surface."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from src.services.models.triton_control import TritonControlService
from src.utils.upstream_errors import UpstreamError


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


@pytest.mark.asyncio
async def test_an_unreachable_repository_index_raises_not_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = _patch(monkeypatch, httpx.ConnectError('down'))
    svc.http_url = 'http://t'
    with pytest.raises(UpstreamError) as caught:
        await svc.get_repository_index()
    assert caught.value.reason == 'triton_unavailable'


class _Reply:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                'bad',
                request=httpx.Request('POST', 'http://t'),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'reply', [_Reply(503, []), _Reply(200, ValueError('not json'))], ids=['non_2xx', 'bad_body']
)
async def test_a_failed_or_unreadable_index_reply_raises(
    monkeypatch: pytest.MonkeyPatch, reply: _Reply
) -> None:
    class _Ok(_Client):
        async def get(self, *_a: Any, **_k: Any) -> Any:
            return reply

        post = get

    monkeypatch.setattr(httpx, 'AsyncClient', lambda **_k: _Ok(RuntimeError()))
    svc = TritonControlService.__new__(TritonControlService)
    svc.http_url = 'http://t'
    with pytest.raises(UpstreamError):
        await svc.get_repository_index()


@pytest.mark.asyncio
async def test_a_good_index_reply_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{'name': 'm', 'state': 'READY'}]

    class _Ok(_Client):
        async def get(self, *_a: Any, **_k: Any) -> Any:
            return _Reply(200, rows)

        post = get

    monkeypatch.setattr(httpx, 'AsyncClient', lambda **_k: _Ok(RuntimeError()))
    svc = TritonControlService.__new__(TritonControlService)
    svc.http_url = 'http://t'
    assert await svc.get_repository_index() == rows


@pytest.mark.asyncio
async def test_a_bug_is_not_reported_as_triton_down(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _patch(monkeypatch, RuntimeError('bug'))
    svc.http_url = 'http://t'
    with pytest.raises(RuntimeError):
        await svc.server_ready()
    with pytest.raises(RuntimeError):
        await svc.get_repository_index()
