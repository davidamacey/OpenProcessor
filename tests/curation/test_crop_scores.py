"""Tests for the crop_scores/ overlay scorers (curation-strategy plan §9).

Pure-math tests against synthetic embeddings — no OpenSearch, no faiss
index files on disk. ``compute_uniqueness`` / ``compute_near_dup_groups``
are the testable cores; the ``*Scorer`` classes are thin OpenSearch-facing
wrappers exercised separately in ``test_scores_router.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.services.curation.item_scores import SCORER_METADATA, available_scorers, get_scorer
from src.services.curation.item_scores.near_dup import NearDupScorer, compute_near_dup_groups
from src.services.curation.item_scores.uniqueness import compute_uniqueness


# =============================================================================
# Uniqueness — 3 tight gaussian clusters + 5 planted far outliers
# =============================================================================


def _make_clustered_embeddings_with_outliers(
    seed: int = 0, dim: int = 32, n_per_cluster: int = 40, n_outliers: int = 5
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """3 tight clusters (~120 points) + n_outliers points planted far from
    every cluster centroid. Returns ``(embeddings, outlier_row_indices,
    cluster_centers)`` — all unit-norm."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    cluster_rows = []
    for c in centers:
        pts = c[None, :] + rng.normal(scale=0.01, size=(n_per_cluster, dim))
        cluster_rows.append(pts)
    cluster_mat = np.vstack(cluster_rows)

    outliers: list[np.ndarray] = []
    while len(outliers) < n_outliers:
        v = rng.normal(size=dim)
        v /= np.linalg.norm(v)
        # Require the candidate to be far (cosine distance > 0.9) from every
        # cluster center so it's unambiguously an outlier, not just unlucky
        # noise on the edge of a cluster.
        if min(1.0 - float(v @ c) for c in centers) > 0.9:
            outliers.append(v)
    outlier_mat = np.vstack(outliers)

    embeddings = np.vstack([cluster_mat, outlier_mat]).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    outlier_idx = list(range(len(cluster_mat), len(cluster_mat) + n_outliers))
    return embeddings, outlier_idx, centers.astype(np.float32)


def test_uniqueness_ranks_planted_outliers_top5() -> None:
    embeddings, outlier_idx, centers = _make_clustered_embeddings_with_outliers()
    scores = compute_uniqueness(embeddings, centers, k=10, nprobe=12)

    assert scores.shape == (embeddings.shape[0],)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0 + 1e-6

    order = np.argsort(-scores)
    top5 = set(order[:5].tolist())
    assert top5 == set(outlier_idx), f'expected outliers {outlier_idx} in top5, got {sorted(top5)}'


def test_uniqueness_degenerate_inputs_return_zeros() -> None:
    assert compute_uniqueness(
        np.zeros((0, 8), dtype=np.float32), np.zeros((2, 8), dtype=np.float32)
    ).shape == (0,)
    single = np.ones((1, 8), dtype=np.float32)
    out = compute_uniqueness(single, single, k=5, nprobe=4)
    assert out.shape == (1,)
    assert out[0] == 0.0


# =============================================================================
# Near-duplicate — exact/near-identical rows group; distinct rows don't
# =============================================================================


def test_near_dup_groups_exact_duplicates_only() -> None:
    rng = np.random.default_rng(1)
    dim = 16
    base = rng.normal(size=(8, dim))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    # Sanity: random high-dim unit vectors are (with overwhelming
    # probability) far below the 0.98 dup threshold pairwise.
    sims = base @ base.T
    np.fill_diagonal(sims, -1.0)
    assert sims.max() < 0.98, f'synthetic base vectors are not well separated: max sim {sims.max()}'

    dup = base[0] + rng.normal(scale=1e-6, size=dim)
    dup /= np.linalg.norm(dup)
    embeddings = np.vstack([base, dup]).astype(np.float32)

    groups = compute_near_dup_groups(embeddings, threshold=0.98)
    assert len(groups) == 1
    assert set(groups[0]) == {0, 8}


def test_near_dup_no_groups_when_all_distinct() -> None:
    rng = np.random.default_rng(2)
    embeddings = rng.normal(size=(10, 16)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    groups = compute_near_dup_groups(embeddings, threshold=0.98)
    assert groups == []


def test_near_dup_bucket_scoping_matches_global_for_small_pools() -> None:
    """When centroids are supplied but nlist >= n, bucket-scoping degrades
    to a single global pass (no bucket boundary should split a real dup
    pair) — regression guard for the O(n^2)-avoidance code path."""
    rng = np.random.default_rng(3)
    dim = 16
    base = rng.normal(size=(6, dim))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    dup = base[2] + rng.normal(scale=1e-6, size=dim)
    dup /= np.linalg.norm(dup)
    embeddings = np.vstack([base, dup]).astype(np.float32)

    global_groups = compute_near_dup_groups(embeddings, threshold=0.98)
    # centroids.shape[0] >= n short-circuits to the global path in
    # compute_near_dup_groups, matching production's small-pool behaviour.
    centroids = rng.normal(size=(embeddings.shape[0] + 2, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    bucketed_groups = compute_near_dup_groups(embeddings, threshold=0.98, centroids=centroids)
    assert bucketed_groups == global_groups


def test_near_dup_raises_when_centroid_store_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``NearDupScorer._load_centroids`` must fail loudly when the persisted
    IVF centroid store is missing, not silently degrade to an unscoped
    global O(n^2) pass over the full residual pool (plan §3 compute budget —
    the exact cost bucket-scoping exists to avoid)."""
    from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore

    monkeypatch.setattr(IVFCentroidStore, 'load', lambda _self: False)
    scorer = NearDupScorer()
    with pytest.raises(RuntimeError, match='IVF centroid store'):
        scorer._load_centroids()


# =============================================================================
# Regression guard — no scorer ever writes a cluster field (plan §8 #3)
# =============================================================================


_FORBIDDEN_CLUSTER_FIELDS = frozenset({'cluster_id', 'cluster_subid', 'cluster_distance'})


@pytest.mark.parametrize('scorer_name', available_scorers())
def test_no_scorer_writes_cluster_fields(scorer_name: str) -> None:
    scorer = get_scorer(scorer_name)
    writes = set(scorer.writes)
    assert not (writes & _FORBIDDEN_CLUSTER_FIELDS), (
        f'{scorer_name} scorer writes forbidden cluster field(s): {writes & _FORBIDDEN_CLUSTER_FIELDS}'
    )


def test_scorer_metadata_writes_match_scorer_class_writes() -> None:
    """SCORER_METADATA (feeds /legacy/methods) must not drift from the actual
    CropScorer.writes ClassVar."""
    for name in available_scorers():
        scorer = get_scorer(name)
        assert set(SCORER_METADATA[name]['writes']) == set(scorer.writes)
        assert not (set(SCORER_METADATA[name]['writes']) & _FORBIDDEN_CLUSTER_FIELDS)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
