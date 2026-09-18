"""Unit tests for the clustering orchestrator's AHC residual path.

All heavy IO is mocked. Synthetic embeddings stand in for the
foundation-model / classifier embeddings so we get well-separated
gaussians the clusterer can resolve in < 1s.

This is the "distinct file, same basename as tests/test_legacy_clustering.py"
file from the reference tree's clustering-package test module — see
``tests/curation/test_clustering_orchestrator_extra.py`` for the
other one.

Deviation from the reference file: the reference version also exercises
a standalone CLI clustering-eval bench/eval harness script via
``compute_metrics``. That script is never scheduled for porting
anywhere in the genericization plan (not in §1's target layout, not in
Appendix A future scope) — it is reference-line-only tooling, out of
scope for this wave. Those two eval-harness tests are dropped here;
only the ``cluster_residuals``-through-AHC test, which exercises
ported orchestrator code, is kept.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def synthetic_embeddings() -> dict[str, Any]:
    """50 crops drawn from 3 well-separated 16-d gaussians.

    Returns embedding matrix + per-crop VLM label aligned with the
    generating gaussian (so coherence is trivially perfect).
    """
    rng = np.random.default_rng(0)
    dim = 16
    centers = np.stack(
        [
            np.concatenate([np.array([5.0]), np.zeros(dim - 1)]),
            np.concatenate([np.zeros(dim // 2), np.array([5.0]), np.zeros(dim - dim // 2 - 1)]),
            np.concatenate([np.zeros(dim - 1), np.array([5.0])]),
        ]
    )
    sizes = [20, 18, 12]
    chunks: list[np.ndarray] = []
    labels: list[str] = []
    label_names = ['carA', 'truckB', 'bikeC']
    for c, n, name in zip(centers, sizes, label_names, strict=True):
        chunks.append(c + 0.05 * rng.standard_normal((n, dim)).astype(np.float32))
        labels.extend([name] * n)
    emb = np.vstack(chunks).astype(np.float32)
    return {'embeddings': emb, 'vlm_labels': labels}


def _make_os_client_with(embeddings: np.ndarray) -> Any:
    hits = [
        {
            '_id': f'crop-{i}',
            '_source': {'pe_embedding': emb.tolist()},
        }
        for i, emb in enumerate(embeddings)
    ]
    client = MagicMock()
    client.search = AsyncMock(return_value={'_scroll_id': 's1', 'hits': {'hits': hits}})
    client.scroll = AsyncMock(return_value={'_scroll_id': None, 'hits': {'hits': []}})
    client.clear_scroll = AsyncMock(return_value=None)
    client.bulk = AsyncMock(return_value={'errors': False})
    client.indices = MagicMock()
    client.indices.refresh = AsyncMock(return_value=None)
    client.get = AsyncMock(side_effect=Exception('no cached reducer'))
    client.index = AsyncMock(return_value=None)
    return client


def _patch_umap_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace UMAP with identity-ish so tests stay under 1s.

    AHC imports umap lazily inside
    :func:`embedding_reduce.get_or_fit_reducer`. We patch that helper
    to skip UMAP entirely and return the input.
    """
    from src.services.curation.clustering import embedding_reduce

    async def _fake_get_or_fit(
        client: Any,
        embeddings: np.ndarray,
        *,
        mode: str,
    ) -> tuple[Any, np.ndarray, bool]:
        return MagicMock(), embeddings.astype(np.float32), True

    monkeypatch.setattr(embedding_reduce, 'get_or_fit_reducer', _fake_get_or_fit)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cluster_residuals_runs_ahc(
    synthetic_embeddings: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cluster_residuals runs AHC complete+cosine+threshold end-to-end."""
    from src.services.curation.clustering.orchestrator import (
        AHC_DISTANCE_THRESHOLD,
        AHC_LINKAGE,
        AHC_METRIC,
        cluster_residuals,
    )

    _patch_umap_passthrough(monkeypatch)
    client = _make_os_client_with(synthetic_embeddings['embeddings'])

    # IVF is the default method since the FAISS-IVF backend landed; request
    # AHC explicitly so this end-to-end test still exercises the AHC path.
    res = await cluster_residuals(client, clustering_method='ahc')
    assert res['method'] == 'ahc'
    # Method-specific knobs now live under cluster_method_params (the result
    # envelope was unified across ClusterMethods).
    params = res['cluster_method_params']
    assert params['linkage'] == AHC_LINKAGE
    assert params['metric'] == AHC_METRIC
    assert params['distance_threshold'] == AHC_DISTANCE_THRESHOLD
    assert res['status'] == 'success'
    assert res['n_residuals'] == 50
    assert res['n_clusters'] >= 2
    assert res['n_noise'] == 0
