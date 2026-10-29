"""Issue #356: the per-project index bootstrap (exists checks + mapping
migrations, 380-470 ms cold) must run at worker startup, off the request
path, so a recycled uvicorn worker's first request does not pay it.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

from src.config import get_curation_config
from src.core import dependencies
from src.services.curation import index_bootstrap


@pytest.mark.asyncio
async def test_startup_runs_full_bootstrap_for_every_active_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound: list[str] = []
    slugs = ['alpha', 'beta']

    def fake_for_each_project(*_a: object, **_kw: object):
        for slug in slugs:
            bound.append(slug)
            yield slug

    async def fake_ensure_indexes(_os: object) -> None:
        index_bootstrap._INDEXES_BOOTSTRAPPED.add(bound[-1])

    wrapper = AsyncMock()
    wrapper.client = AsyncMock()
    monkeypatch.setattr(
        dependencies.OpenSearchClientFactory, 'get_client', AsyncMock(return_value=wrapper)
    )
    monkeypatch.setattr('src.services.projects.bootstrap.for_each_project', fake_for_each_project)
    monkeypatch.setattr(index_bootstrap, 'ensure_indexes', fake_ensure_indexes)
    monkeypatch.setattr(index_bootstrap, 'warm_knn_indexes', AsyncMock())
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', set())

    task = await dependencies.bootstrap_opensearch_indexes()
    if task is not None:
        await task

    assert {'alpha', 'beta'} == index_bootstrap._INDEXES_BOOTSTRAPPED


@pytest.mark.asyncio
async def test_warmed_project_request_does_no_opensearch_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        index_bootstrap, '_INDEXES_BOOTSTRAPPED', {get_curation_config().project_slug}
    )
    client = AsyncMock()
    await index_bootstrap.ensure_indexes(client)
    assert client.method_calls == []


def test_forget_bootstrapped_makes_next_request_rebootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', {'gone', 'kept'})
    index_bootstrap.forget_bootstrapped('gone')
    assert {'kept'} == index_bootstrap._INDEXES_BOOTSTRAPPED
    index_bootstrap.forget_bootstrapped('never-seen')  # no error


def test_text_status_labels_cache_as_per_process(tmp_path) -> None:
    from src.clients.pe_encoder import PEEncoder

    status = PEEncoder(text_onnx_path=tmp_path / 'x.onnx').text_status()
    assert status['cache_scope'] == 'process'
    assert status['pid'] == os.getpid()


def test_prime_text_backend_hits_backend_without_touching_cache(tmp_path) -> None:
    import numpy as np

    from src.clients.pe_encoder import PEEncoder

    calls: list[int] = []

    class _Backend:
        name = 'triton'

        def encode(self, tokens):
            calls.append(len(tokens))
            return np.ones((len(tokens), 1024), dtype=np.float32)

    enc = PEEncoder(text_onnx_path=tmp_path / 'x.onnx')
    enc._text_tokenizer = lambda qs: np.zeros((len(qs), 77), dtype=np.int64)
    enc._text_backend = _Backend()
    enc._text_ready = True

    enc.prime_text_backend()

    assert calls == [1]
    info = enc.text_cache_info()
    assert (info.hits, info.misses, info.currsize) == (0, 0, 0)


def test_prime_text_backend_skips_lazy_local_backends(tmp_path) -> None:
    from src.clients.pe_encoder import PEEncoder

    class _Lazy:
        name = 'torch'

        def encode(self, _tokens):
            raise AssertionError('must not load a local model at startup')

    enc = PEEncoder(text_onnx_path=tmp_path / 'x.onnx')
    enc._text_backend = _Lazy()
    enc._text_ready = True
    enc.prime_text_backend()
