"""A promoted model an operator unloaded stays unloaded: the periodic reload
tick (and the startup reload) skip it; an explicit load, a promote or the
explicit reload route bring it back."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest

from src.services.training import triton_promote
from src.services.training.triton_promote import (
    UNLOADED_MARKER,
    TritonPromoter,
    reload_promoted_models,
    set_explicitly_unloaded,
)


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    model = tmp_path / 'cars__det'
    model.mkdir()
    (model / 'promote.json').write_text(json.dumps({'project': 'cars'}))
    return tmp_path


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Triton reports nothing READY; every ``/load`` is recorded and succeeds."""
    posted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/repository/index'):
            return httpx.Response(200, json=[])
        posted.append(request.url.path)
        return httpx.Response(200)

    real = httpx.AsyncClient
    monkeypatch.setattr(
        triton_promote.httpx,
        'AsyncClient',
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    return posted


def _promoter(repo: Path) -> TritonPromoter:
    return TritonPromoter(triton_models_dir=repo, triton_http_url='http://triton')


@pytest.mark.asyncio
async def test_an_unloaded_model_is_not_reloaded_by_the_tick(repo: Path, loads: list[str]) -> None:
    set_explicitly_unloaded(repo / 'cars__det', True)
    for _ in range(2):
        result = await reload_promoted_models(_promoter(repo))
        assert result == {'status': 'ok', 'reloaded': [], 'failed': []}
    assert loads == []


@pytest.mark.asyncio
async def test_a_model_never_unloaded_is_reloaded(repo: Path, loads: list[str]) -> None:
    result = await reload_promoted_models(_promoter(repo))
    assert result['reloaded'] == ['cars__det']
    assert loads == ['/v2/repository/models/cars__det/load']


@pytest.mark.asyncio
async def test_the_explicit_reload_loads_it_and_clears_the_marker(
    repo: Path, loads: list[str]
) -> None:
    set_explicitly_unloaded(repo / 'cars__det', True)
    result = await reload_promoted_models(_promoter(repo), honor_unloaded=False)
    assert result['reloaded'] == ['cars__det']
    assert not (repo / 'cars__det' / UNLOADED_MARKER).exists()
    assert loads


def test_a_directory_without_promote_json_gets_no_marker(tmp_path: Path) -> None:
    plain = tmp_path / 'core_model'
    plain.mkdir()
    set_explicitly_unloaded(plain, True)
    assert not (plain / UNLOADED_MARKER).exists()


def test_the_unload_route_records_the_marker(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from src.routers import models as models_router

    monkeypatch.setattr(models_router, 'TRITON_MODELS_DIR', repo)
    monkeypatch.setattr(models_router, 'check_unload', lambda *_a, **_k: None)

    class _Triton:
        async def unload_model(self, name: str) -> tuple[bool, str]:
            return True, f'{name} unloaded'

        async def load_model(self, name: str) -> tuple[bool, str]:
            return True, f'{name} loaded'

    monkeypatch.setattr(models_router, 'TritonControlService', _Triton)
    from src.main import app

    client = TestClient(app)
    assert client.post('/models/cars__det/unload').status_code == 200
    assert (repo / 'cars__det' / UNLOADED_MARKER).exists()
    assert client.post('/models/cars__det/load').status_code == 200
    assert not (repo / 'cars__det' / UNLOADED_MARKER).exists()
