"""Issue #356: a bootstrapped project's request does no OpenSearch work. The bootstrap itself
now runs once in ``db-migrate`` (``tests/opensearch/``), not at worker startup."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

from src.config import get_curation_config
from src.services.curation import index_bootstrap


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


def test_text_status_labels_cache_as_per_process() -> None:
    from src.clients.pe_encoder import PEEncoder

    status = PEEncoder().text_status()
    assert status['cache_scope'] == 'process'
    assert status['pid'] == os.getpid()


def test_prime_text_backend_hits_backend_without_touching_cache() -> None:
    import numpy as np

    from src.clients.pe_encoder import PEEncoder

    calls: list[int] = []

    class _Backend:
        name = 'triton'

        def encode(self, tokens):
            calls.append(len(tokens))
            return np.ones((len(tokens), 1024), dtype=np.float32)

    enc = PEEncoder()
    enc._text_tokenizer = lambda qs: np.zeros((len(qs), 77), dtype=np.int64)
    enc._text_backend = _Backend()
    enc._text_ready = True

    enc.prime_text_backend()

    assert calls == [1]
    info = enc.text_cache_info()
    assert (info.hits, info.misses, info.currsize) == (0, 0, 0)
