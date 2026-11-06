"""Per-stage timing of the curation kNN search routes (#414): Server-Timing header and the
``op_search_stage_seconds`` histogram."""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.utils.search_timing import SEARCH_STAGES, bind_search_timer, search_annotate, search_span


def _parse(header: str) -> dict[str, float]:
    return {p.split(';dur=')[0]: float(p.split(';dur=')[1]) for p in header.split(', ')}


def _encoder(ready: bool = True) -> MagicMock:
    enc = MagicMock()
    enc.text_ready = ready
    enc.encode_text = MagicMock(return_value=np.ones((1, 1024), dtype=np.float32))
    return enc


def _client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    search_s: float = 0,
    count_s: float = 0,
    encoder=None,
    search_hits=None,
) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    monkeypatch.setenv('OP_SEMANTIC_SEARCH_ENABLED', 'true')

    async def _search(**_kw):
        await asyncio.sleep(search_s)
        hits = search_hits or []
        return {'took': 7, 'hits': {'hits': hits, 'total': {'value': len(hits)}}}

    async def _count(**_kw):
        await asyncio.sleep(count_s)
        return {'count': 0}

    fake_os = AsyncMock()
    fake_os.search = _search
    fake_os.count = _count
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.state.pe_encoder = encoder or _encoder()
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


def test_unknown_stage_raises():
    with (
        bind_search_timer('text'),
        pytest.raises(ValueError, match='unknown search stage'),
        search_span('bogus'),
    ):
        pass


def test_helpers_are_noops_without_a_timer():
    with search_span('encode'):
        pass
    search_annotate('encode_cache_hit', 1)


def test_header_reports_overlap_not_sum(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch, search_s=0.05, count_s=0.03)
    resp = client.get('/curation/projects/default/search/text', params={'q': 'truck'})
    assert resp.status_code == 200
    stages = _parse(resp.headers['Server-Timing'])
    assert stages['knn'] >= 50
    assert stages['unembedded_count'] >= 30
    # knn and the count run together: the page span is the longer one, not their sum
    assert stages['knn_page'] < 80
    assert stages['os_took_ms'] == 7
    assert {'encode', 'filter_build', 'hydrate'} <= stages.keys()
    assert set(stages) - {'os_took_ms', 'encode_cache_hit'} <= set(SEARCH_STAGES)


def test_header_absent_on_error(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch, encoder=_encoder(ready=False))
    resp = client.get('/curation/projects/default/search/text', params={'q': 'truck'})
    assert resp.status_code == 503
    assert 'Server-Timing' not in resp.headers


def test_histogram_observed_per_stage(monkeypatch: pytest.MonkeyPatch):
    def count() -> float:
        labels = {'route': 'text', 'stage': 'knn'}
        return REGISTRY.get_sample_value('op_search_stage_seconds_count', labels) or 0.0

    before = count()
    _client(monkeypatch).get('/curation/projects/default/search/text', params={'q': 'truck'})
    assert count() == before + 1


def test_similar_reports_source_vector(monkeypatch: pytest.MonkeyPatch):
    hit = {'_source': {'pe_embedding': [0.1] * 4}}
    client = _client(monkeypatch, search_hits=[hit])
    resp = client.get('/curation/projects/default/search/similar', params={'crop_id': 'c1'})
    assert resp.status_code == 200, resp.text
    assert 'source_vector' in _parse(resp.headers['Server-Timing'])


def test_encode_text_annotates_cache_hit():
    from src.clients import pe_encoder

    enc = pe_encoder.PEEncoder.__new__(pe_encoder.PEEncoder)
    enc._cache_lock = threading.Lock()
    enc._text_cache = OrderedDict()
    enc._cache_hits = enc._cache_misses = 0
    enc._encode_uncached = lambda qs: np.ones((len(qs), 1024), dtype=np.float32)  # type: ignore[method-assign]
    with bind_search_timer('text') as timer:
        enc.encode_text(['a'])
        assert timer.annotations['encode_cache_hit'] == 0
        enc.encode_text(['a'])
        assert timer.annotations['encode_cache_hit'] == 1
