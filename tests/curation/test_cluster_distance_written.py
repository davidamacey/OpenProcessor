"""Every residual clustering run writes ``cluster_distance``.

The review queue's outlier/representativeness sorts and the cluster view's
core-member cut line (``cluster_similarity = 1 - cluster_distance``) read
this field. IVF's small-pool single-bucket fallback (fewer than
``IVF_MIN_TRAIN_VECTORS`` residuals — every small dataset) and the
label-only methods (AHC, HDBSCAN) returned no distances, so the
orchestrator wrote ``cluster_distance: null`` on every item and those
features were dead. The orchestrator now falls back to the cosine
distance to the member-mean centroid of each item's cluster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


if TYPE_CHECKING:
    from pathlib import Path


def _embeddings() -> np.ndarray:
    rng = np.random.default_rng(0)
    dim = 16
    centers = np.eye(dim, dtype=np.float32)[[0, 8, 15]] * 5.0
    chunks = [
        c + 0.05 * rng.standard_normal((n, dim))
        for c, n in zip(centers, (20, 18, 12), strict=False)
    ]
    emb = np.vstack(chunks).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def _client() -> Any:
    client = MagicMock()
    client.bulk = AsyncMock(return_value={'errors': False})
    client.indices = MagicMock()
    client.indices.refresh = AsyncMock(return_value=None)
    client.get = AsyncMock(side_effect=Exception('no cached reducer'))
    client.index = AsyncMock(return_value=None)
    return client


def _written(client: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for call in client.bulk.await_args_list:
        body = call.kwargs['body']
        for action, doc in zip(body[::2], body[1::2], strict=True):
            # F-3: residual writes are guarded painless scripts; the written
            # values travel as script params.
            params = doc['script']['params']
            out[action['update']['_id']] = {
                'cluster_id': params['cid'],
                'cluster_distance': params['dist'],
            }
    return out


@pytest.fixture
def residuals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> np.ndarray:
    from src.services.curation.clustering import embedding_reduce
    from src.services.curation.clustering.methods import ivf_store

    emb = _embeddings()
    ids = [f'crop-{i}' for i in range(len(emb))]
    monkeypatch.setattr(
        embedding_reduce,
        'fetch_residual_embeddings_parallel',
        AsyncMock(return_value=(ids, emb)),
    )

    async def _identity_reducer(_client: Any, x: np.ndarray, *, mode: str) -> Any:
        return MagicMock(), x.astype(np.float32), True

    monkeypatch.setattr(embedding_reduce, 'get_or_fit_reducer', _identity_reducer)
    monkeypatch.setattr(ivf_store, 'IVF_STORE_DIR', tmp_path / 'ivf')
    return emb


def _expected(emb: np.ndarray, members: list[int]) -> np.ndarray:
    centroid = emb[members].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    return 1.0 - emb[members] @ centroid


@pytest.mark.asyncio
async def test_ivf_single_bucket_fallback_writes_centroid_distance(residuals: np.ndarray) -> None:
    from src.services.curation.clustering.orchestrator import cluster_residuals

    client = _client()
    res = await cluster_residuals(client, clustering_method='ivf')
    assert res['cluster_method_extra']['reason'] == 'too_few_vectors_for_training'

    written = _written(client)
    assert len(written) == len(residuals)
    dists = np.array([written[f'crop-{i}']['cluster_distance'] for i in range(len(residuals))])
    np.testing.assert_allclose(dists, _expected(residuals, list(range(len(residuals)))), atol=1e-5)
    # Three separated blobs in one bucket: real spread, not a constant.
    assert dists.max() - dists.min() > 0.1


@pytest.mark.asyncio
async def test_label_only_method_writes_member_centroid_distance(residuals: np.ndarray) -> None:
    from src.services.curation.clustering.orchestrator import cluster_residuals

    client = _client()
    res = await cluster_residuals(client, clustering_method='ahc')
    assert res['n_clusters'] >= 2
    written = _written(client)
    by_cluster: dict[int, list[int]] = {}
    for i in range(len(residuals)):
        by_cluster.setdefault(written[f'crop-{i}']['cluster_id'], []).append(i)
    for members in by_cluster.values():
        got = [written[f'crop-{i}']['cluster_distance'] for i in members]
        np.testing.assert_allclose(got, _expected(residuals, members), atol=1e-5)


def test_member_centroid_distances_leave_noise_empty() -> None:
    from src.services.curation.clustering.centroid_distance import member_centroid_distances

    emb = _embeddings()[:6]
    labels = np.array([0, 0, 0, -1, 1, 1])
    out = member_centroid_distances(emb, labels)
    assert out[3] is None
    assert all(isinstance(d, float) and 0.0 <= d <= 2.0 for i, d in enumerate(out) if i != 3)
