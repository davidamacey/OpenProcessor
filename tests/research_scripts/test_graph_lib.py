"""Graph construction and CHAMELEON merge on a planted partition (no FAISS, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research import graph_lib as gl
from scripts.research.planted import planted


def _blobs(per_class: int = 60, classes: int = 3, dim: int = 24, seed: int = 3):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((classes, dim))
    truth = np.repeat(np.arange(classes), per_class)
    x = centres[truth] + 0.35 * rng.standard_normal((len(truth), dim))
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    return x, truth


def _exact_knn(x: np.ndarray, k: int):
    sims = x @ x.T
    np.fill_diagonal(sims, -2.0)
    idx = np.argsort(-sims, axis=1)[:, :k]
    return np.take_along_axis(sims, idx, axis=1), idx


def test_mutual_graph_is_symmetric_and_stays_inside_planted_classes():
    x, truth = _blobs()
    graph = gl.mutual_graph(*_exact_knn(x, 10))
    assert abs(graph - graph.T).max() == 0
    r, c, w = gl.upper_edges(graph)
    assert len(r) > 0
    assert (truth[r] == truth[c]).all()
    assert w.min() > 0


def test_mutual_graph_drops_one_way_neighbours():
    sims = np.array([[0.9, 0.8], [0.9, 0.7], [0.8, 0.7]], dtype=np.float32)
    idx = np.array([[1, 2], [0, 2], [0, 0]])  # node 2 lists 0 but not 1; 1 lists 2 one-way
    graph = gl.mutual_graph(sims, idx)
    assert graph[0, 1] > 0
    assert graph[0, 2] > 0
    assert graph[1, 2] == 0


def test_chameleon_merges_oversplit_class_but_never_two_classes():
    x, truth = _blobs()
    graph = gl.mutual_graph(*_exact_knn(x, 12))
    rng = np.random.default_rng(0)
    atoms = gl.codes(truth, rng.integers(0, 4, len(truth)))  # each class cut into 4 pieces
    merged = gl.chameleon_merge(graph, atoms, ri_min=0.2, rc_min=0.8)
    assert merged.max() + 1 <= atoms.max() + 1
    for cluster in np.unique(merged):
        assert len(np.unique(truth[merged == cluster])) == 1  # no false merge
    assert merged.max() + 1 == 3  # all pieces of a class rejoined


def test_chameleon_refuses_to_merge_when_floors_are_unreachable():
    x, truth = _blobs()
    graph = gl.mutual_graph(*_exact_knn(x, 12))
    atoms = gl.codes(truth, np.arange(len(truth)) % 2)
    kept = gl.chameleon_merge(graph, atoms, ri_min=50.0, rc_min=0.0)
    assert (kept == np.unique(atoms, return_inverse=True)[1]).all()


def test_min_size_turns_small_clusters_into_unassigned():
    out = gl.apply_min_size(np.array([0, 0, 0, 1, 2, 2, 2, 2]), 3)
    assert out.tolist() == [0, 0, 0, -1, 1, 1, 1, 1]


def test_leiden_cpm_recovers_planted_classes_when_available():
    pytest.importorskip('leidenalg')
    x, truth = _blobs()
    graph = gl.mutual_graph(*_exact_knn(x, 12))
    labels = gl.leiden_cpm(graph, 0.05, seed=1)
    for cluster in np.unique(labels):
        assert len(np.unique(truth[labels == cluster])) == 1


def test_planted_generator_is_seeded_and_labels_noise_as_singletons():
    a, la = planted(2000, dim=32, seed=5)
    b, lb = planted(2000, dim=32, seed=5)
    assert (a == b).all()
    assert (la == lb).all()
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    noise = [name for name in la.tolist() if name.startswith('noise')]
    assert len(noise) == len(set(noise)) == 100
