"""The callers that opened their own fixed-URL pool now follow the serving-profile router."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.clients.pe_encoder import PE_EMBEDDING_DIM, PEEncoder, TextModelNotReady
from src.clients.pe_text_backends import PE_TEXT_TRITON_MODEL
from src.clients.triton_client_manager import TritonClientManager
from src.clients.triton_router import TritonRouter
from src.services import serving_router


class _Result:
    def as_numpy(self, name: str) -> np.ndarray:
        return np.ones((1, PE_EMBEDDING_DIM), dtype=np.float32)


class _Server:
    def __init__(self) -> None:
        self.infers = 0

    def is_model_ready(self, model_name: str) -> bool:
        return True

    def infer(self, *_a: Any, **_k: Any) -> _Result:
        self.infers += 1
        return _Result()


def _router(ready: dict[str, list[str]]) -> TritonRouter:
    router = TritonRouter({'a': 'a:8001', 'b': 'b:8001'})
    router.update_ready(ready)
    return router


def _encoder(
    monkeypatch: pytest.MonkeyPatch, router: TritonRouter
) -> tuple[PEEncoder, SimpleNamespace, dict[str, _Server]]:
    servers = {'a:8001': _Server(), 'b:8001': _Server()}
    monkeypatch.setattr(TritonClientManager, 'get_sync_client', lambda url, **_k: servers[url])
    pool = SimpleNamespace(router=router)
    enc = PEEncoder(triton_pool=pool)  # type: ignore[arg-type]
    monkeypatch.setattr(
        enc, '_load_tokenizer', lambda: lambda qs: np.full((len(qs), 4), 7, dtype=np.int64)
    )
    return enc, pool, servers


def test_the_text_encoder_calls_the_server_the_router_picks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enc, _, servers = _encoder(monkeypatch, _router({'b': [PE_TEXT_TRITON_MODEL]}))
    enc.warm_text_encoder()
    enc.encode_text(['white truck'])
    assert (servers['a:8001'].infers, servers['b:8001'].infers) == (0, 1)


def test_the_text_encoder_follows_a_router_the_pool_adopted_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enc, pool, servers = _encoder(monkeypatch, _router({'b': [PE_TEXT_TRITON_MODEL]}))
    enc.warm_text_encoder()
    enc.encode_text(['white truck'])
    pool.router = _router({'a': [PE_TEXT_TRITON_MODEL]})
    enc.encode_text(['red bus'])
    assert (servers['a:8001'].infers, servers['b:8001'].infers) == (1, 1)


def test_no_server_reporting_the_text_model_means_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enc, _, _ = _encoder(monkeypatch, _router({'a': ['other']}))
    with pytest.raises(TextModelNotReady):
        enc.warm_text_encoder()


@pytest.fixture
def routed(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every pool a caller asks the serving router for (the pool itself is a stub)."""
    asked: list[dict[str, Any]] = []

    class _Pool:
        router = TritonRouter.single('x:8001')

        async def close(self) -> None:
            return None

    async def open_routed_pool(**kwargs: Any) -> _Pool:
        asked.append(kwargs)
        return _Pool()

    monkeypatch.setattr(serving_router, 'open_routed_pool', open_routed_pool)
    return asked


@pytest.mark.asyncio
async def test_the_import_job_opens_a_routed_pool(routed: list[dict[str, Any]]) -> None:
    from src.services.curation.dataset_import.job import open_ingest_runtime

    runtime = await open_ingest_runtime()
    assert routed == [{'pool_size': 2, 'max_concurrent': 16}]
    assert runtime.pe_encoder.triton_pool is runtime.pool


@pytest.mark.asyncio
async def test_the_reindex_encoder_opens_a_routed_pool(
    monkeypatch: pytest.MonkeyPatch, routed: list[dict[str, Any]]
) -> None:
    from src.tasks import embedding_reindex

    async def opensearch(_timeout: int) -> Any:
        return SimpleNamespace()

    monkeypatch.setattr(embedding_reindex, 'opensearch_client', opensearch)
    await embedding_reindex._encoder()
    assert routed == [{'pool_size': 4, 'max_concurrent': 64}]


@pytest.mark.asyncio
async def test_the_embed_missing_stage_opens_a_routed_pool(routed: list[dict[str, Any]]) -> None:
    from src.services.curation.autolabel.embed_stage import _owned_encoder

    pool, encoder = await _owned_encoder()
    assert routed == [{'pool_size': 1, 'max_concurrent': 8}]
    assert encoder.triton_pool is pool
