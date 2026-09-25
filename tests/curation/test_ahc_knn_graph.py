"""GPU/CPU kNN-graph parity for the AHC residual method.

cuML's ``kneighbors_graph`` includes each point as its own nearest
neighbor (``n_neighbors=k+1`` requests k *other* points plus self); the
CPU (sklearn) path passes ``include_self=False`` instead. Left in, the
self-loop thickens the GPU graph's connectivity relative to the CPU
graph for the same ``k`` and the same embeddings. ``_gpu_knn`` must drop
the diagonal so both backends produce the same connectivity shape.

cuML itself isn't installed in this environment (the GPU AHC path is
inactive on the box the audit ran against), so ``cuml.neighbors`` is
stubbed via ``sys.modules`` with a fake ``NearestNeighbors`` that
reproduces the exact self-inclusive shape cuML returns.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from src.services.curation.clustering.backend import BackendInfo
from src.services.curation.clustering.methods.ahc import _build_knn_graph


class _FakeCuNearestNeighbors:
    """Reproduces cuML's kneighbors_graph: self-inclusive connectivity."""

    def __init__(self, n_neighbors: int, metric: str) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, embeddings: np.ndarray) -> None:
        self._n = embeddings.shape[0]

    def kneighbors_graph(self, embeddings: np.ndarray, mode: str) -> Any:  # noqa: ARG002
        n = embeddings.shape[0]
        # Every point's row includes itself (diagonal = 1) plus one
        # arbitrary neighbor -- enough to prove the diagonal gets
        # stripped without needing real nearest-neighbor math.
        dense = np.eye(n, dtype=np.float64)
        for i in range(n):
            dense[i, (i + 1) % n] = 1.0
        return sp.csr_matrix(dense)


@pytest.fixture
def fake_cuml_neighbors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType('cuml.neighbors')
    fake_module.NearestNeighbors = _FakeCuNearestNeighbors  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'cuml.neighbors', fake_module)
    monkeypatch.setitem(sys.modules, 'cuml', types.ModuleType('cuml'))


@pytest.mark.asyncio
async def test_gpu_knn_graph_has_no_self_loops(fake_cuml_neighbors: None) -> None:
    embeddings = np.random.default_rng(0).normal(size=(6, 8)).astype(np.float32)
    backend_info = BackendInfo(name='gpu', detail='fake', free_vram_mb=None, build_algo=None)

    graph, backend_used = await _build_knn_graph(embeddings, k=2, backend_info=backend_info)

    assert backend_used == 'cuml'
    dense = np.asarray(graph.todense()) if hasattr(graph, 'todense') else np.asarray(graph)
    assert np.all(np.diagonal(dense) == 0), 'GPU kNN graph must not have self-loops'


@pytest.mark.asyncio
async def test_gpu_knn_falls_back_to_cpu_and_also_has_no_self_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the GPU path fails outright, the CPU fallback (which already
    passes include_self=False) is used -- confirms parity holds however
    the graph was built.
    """
    embeddings = np.random.default_rng(1).normal(size=(6, 8)).astype(np.float32)
    backend_info = BackendInfo(name='gpu', detail='fake', free_vram_mb=None, build_algo=None)

    graph, backend_used = await _build_knn_graph(embeddings, k=2, backend_info=backend_info)

    assert backend_used == 'sklearn'
    dense = np.asarray(graph.todense()) if hasattr(graph, 'todense') else np.asarray(graph)
    assert np.all(np.diagonal(dense) == 0)
