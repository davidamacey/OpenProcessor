"""UMAP refit on tiny pools: clamp + random init, typed refusal below the floor."""

from __future__ import annotations

import asyncio
import warnings

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.dependencies import get_curation_opensearch
from src.routers import curation_umap
from src.services.curation.clustering import embedding_reduce as er, pool_size as ps


def _fit(n: int) -> np.ndarray:
    import umap

    x = np.random.RandomState(0).randn(n, 64).astype('float32')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return umap.UMAP(
            min_dist=0.0, metric='cosine', random_state=42, **ps.umap_shape(n)
        ).fit_transform(x)


@pytest.mark.parametrize('n', [ps.MIN_RESIDUALS_FOR_CLUSTERING, 46, 52, 60, 80])
def test_cpu_fit_succeeds_at_small_n(n):
    assert _fit(n).shape[0] == n


def test_shape_clamps_below_n():
    shape = ps.umap_shape(46)
    assert shape['n_components'] < 46
    assert shape['n_neighbors'] < 46
    assert shape['init'] == 'random'
    assert ps.umap_shape(5000) == {'n_components': 50, 'n_neighbors': 15, 'init': 'spectral'}


def test_require_min_items():
    ps.require_min_items(ps.MIN_RESIDUALS_FOR_CLUSTERING)
    with pytest.raises(ps.TooFewItemsError) as exc:
        ps.require_min_items(ps.MIN_RESIDUALS_FOR_CLUSTERING - 1)
    assert exc.value.min_items == ps.MIN_RESIDUALS_FOR_CLUSTERING


def test_rebuild_route_returns_typed_422(monkeypatch):
    async def _fetch(_client):
        return ['a'] * 5, np.zeros((5, 8), dtype='float32')

    monkeypatch.setattr(er, 'fetch_residual_embeddings', _fetch)
    app = FastAPI()
    app.include_router(curation_umap.router)
    app.dependency_overrides[get_curation_opensearch] = lambda: object()
    resp = TestClient(app).post('/cluster/umap/rebuild')
    assert resp.status_code == 422
    assert resp.json()['detail']['error'] == 'too_few_items'
    assert resp.json()['detail']['min_items'] == ps.MIN_RESIDUALS_FOR_CLUSTERING


def test_rebuild_fits_above_floor(monkeypatch):
    n = 46
    x = np.random.RandomState(1).randn(n, 64).astype('float32')

    async def _fetch(_client):
        return [str(i) for i in range(n)], x

    monkeypatch.setattr(er, 'fetch_residual_embeddings', _fetch)
    monkeypatch.setattr(er, '_save_umap_state_to_disk', lambda _reducer, _path: None)

    async def _no_os(_client, _reducer, *, state_id):
        return None

    monkeypatch.setattr(er, '_save_umap_state_to_opensearch', _no_os)
    out = asyncio.run(er.umap_rebuild(object()))
    assert out['status'] == 'success'
    assert out['n_residuals'] == n
