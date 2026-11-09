"""``op.periodic.reload_promoted``: one tick re-loads the promoted models Triton lost and does not
fail on an unreachable Triton (it reports it in the result and the next tick tries again)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from src.services.periodic import reload_promoted
from src.services.training import model_registry, triton_reload
from src.services.training.triton_promote import TritonPromoter


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def promoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'yolo_custom').mkdir()

    async def live() -> list[Any]:
        return [SimpleNamespace(promotion=SimpleNamespace(triton_model_name='yolo_custom'))]

    monkeypatch.setattr(model_registry, 'all_live', live)
    promoter = TritonPromoter(triton_models_dir=tmp_path, triton_http_url='http://triton:8000')
    monkeypatch.setattr(triton_reload, 'TritonPromoter', lambda: promoter)


def triton(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[str]:
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return handler(request)

    transport = httpx.MockTransport(respond)
    real = httpx.AsyncClient
    monkeypatch.setattr(
        triton_reload.httpx, 'AsyncClient', lambda **kw: real(transport=transport, **kw)
    )
    return seen


def test_a_model_triton_lost_is_loaded_again(promoted, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == '/v2/repository/index':
            return httpx.Response(200, json=[{'name': 'yolo_custom', 'state': 'UNAVAILABLE'}])
        return httpx.Response(200)

    seen = triton(monkeypatch, handler)

    result = asyncio.run(reload_promoted.tick())

    assert result['reloaded'] == ['yolo_custom']
    assert '/v2/repository/models/yolo_custom/load' in seen


def test_a_ready_model_is_left_alone(promoted, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = triton(
        monkeypatch,
        lambda _r: httpx.Response(200, json=[{'name': 'yolo_custom', 'state': 'READY'}]),
    )
    result = asyncio.run(reload_promoted.tick())
    assert result == {'status': 'ok', 'reloaded': [], 'failed': []}
    assert seen == ['/v2/repository/index']


def test_an_unreachable_triton_is_a_result_not_a_failed_tick(
    promoted, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('refused', request=request)

    triton(monkeypatch, refuse)

    result = asyncio.run(reload_promoted.tick())

    assert result['status'] == 'error'
    assert result['reloaded'] == []
