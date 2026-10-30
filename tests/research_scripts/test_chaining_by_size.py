"""Y3: the chaining profile on hand-built dendrograms (arithmetic in the comments)."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from scripts.research.cluster_diag import chaining as ch


def _chain() -> np.ndarray:
    # Leaves 0..5; node 6=(0,1) h.1 size2; 7=(6,2) h.2 size3; 8=(7,3) h.3 size4;
    # 9=(8,4) h.4 size5; 10=(9,5) h.5 size6. Every merge after the first adds one crop.
    return np.array(
        [
            [0, 1, 0.1, 2],
            [6, 2, 0.2, 3],
            [7, 3, 0.3, 4],
            [8, 4, 0.4, 5],
            [9, 5, 0.5, 6],
        ],
        dtype=float,
    )


def test_cut_counts_clusters_and_shares() -> None:
    # Cut 0.35 keeps merges 6, 7, 8: one cluster of 4 (node 8) plus leaves 4 and 5 -> 3 clusters;
    # the largest holds 4/6 of the key; clusters of >= 3 hold the same 4 crops.
    rep = ch.chaining_report(_chain(), 0.35)
    assert rep['clusters'] == 3
    assert rep['largest_cluster_crops'] == 4
    assert abs(rep['largest_cluster_share'] - 4 / 6) < 1e-12
    assert abs(rep['share_in_clusters_of_3_or_more'] - 4 / 6) < 1e-12


def test_chain_profile_is_additions_of_one_crop() -> None:
    # Merges of node 8, latest first: (7 size3, leaf) h.3; (6 size2, leaf) h.2; (leaf, leaf) h.1.
    prof = ch.chaining_report(_chain(), 0.35)['largest_cluster_profile']
    assert [p['height'] for p in prof] == [0.3, 0.2, 0.1]
    assert [(p['small_child'], p['large_child']) for p in prof] == [(1, 3), (1, 2), (1, 1)]
    summary = ch.profile_summary(prof)
    assert summary['additions_under_5pct_of_sibling'] == 0  # 1/3 and 1/2 are above 5 %


def test_long_chain_is_flagged_as_additions() -> None:
    # 30 leaves added one by one: each merge's small child is 1/size of the rest, under 5 % once the
    # rest holds more than 20 crops (merges 21..29 -> 9 of them).
    rows, node, size = [], 30, 2
    rows.append([0, 1, 0.01, 2])
    for leaf in range(2, 30):
        rows.append([node, leaf, 0.01 * (leaf), size + 1])
        node, size = node + 1, size + 1
    z = np.array(rows, dtype=float)
    rep = ch.chaining_report(z, 1.0)
    assert rep['largest_cluster_crops'] == 30
    # merges listed: the last 20 (rest sizes 29 down to 10); rest > 20 for sizes 21..29 -> 9
    assert rep['largest_cluster_profile_summary']['additions_under_5pct_of_sibling'] == 9


def test_two_large_children_have_equal_share() -> None:
    z = np.array([[0, 1, 0.1, 2], [2, 3, 0.1, 2], [4, 5, 0.3, 4]], dtype=float)
    top = ch.chaining_report(z, 0.5)['largest_cluster_profile'][0]
    assert (top['small_child'], top['large_child']) == (2, 2)
    assert top['small_share_of_large'] == 1.0


def test_roots_agree_with_scipy_cut() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 8))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    z = linkage(x, method='average', metric='cosine')
    for cut in (0.5, 0.8, 0.95):
        roots = ch.cluster_roots(z, cut)
        sizes = sorted(ch.node_size(z, r) for r in roots)
        counts = sorted(np.bincount(fcluster(z, cut, 'distance'))[1:].tolist())
        assert sizes == counts
