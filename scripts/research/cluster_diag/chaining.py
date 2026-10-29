"""Chaining check by size (Y3): what the clusters of one key look like at a cut, without labels.

A cluster built from two large compact children is a real group. A cluster whose last merges each
add a handful of crops to a big one, at heights just under the cut, is the chaining signature of
average linkage on a dense key. Size is not purity: this only picks what a judge should look at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    Floats = NDArray[np.floating[Any]]

MIN_GROUP = 3  # "crops in clusters of at least 3"
PROFILE_MERGES = 20  # merges of the largest cluster that the profile lists
SMALL_CHILD_SHARE = 0.05  # a child under this share of its sibling counts as an addition


def cluster_roots(z: Floats, cut: float) -> list[int]:
    """Node ids of the clusters at ``cut``: a merge at height <= cut that no later such merge uses.

    Leaves are nodes 0..n-1; merge row ``i`` is node ``n + i``. A leaf that no merge under the cut
    uses is its own cluster.
    """
    n = len(z) + 1
    below = z[:, 2] <= cut
    used = np.zeros(2 * n - 1, dtype=bool)
    used[z[below, 0].astype(np.int64)] = True
    used[z[below, 1].astype(np.int64)] = True
    alive = np.zeros(2 * n - 1, dtype=bool)
    alive[:n] = True
    alive[n:] = below
    return [int(i) for i in np.nonzero(alive & ~used)[0]]


def node_size(z: Floats, node: int) -> int:
    n = len(z) + 1
    return 1 if node < n else int(z[node - n, 3])


def subtree_merges(z: Floats, root: int) -> list[int]:
    """Merge rows inside the subtree of ``root``, most recent (highest row) first."""
    n = len(z) + 1
    rows: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node < n:
            continue
        rows.append(node - n)
        stack.extend((int(z[node - n, 0]), int(z[node - n, 1])))
    return sorted(rows, reverse=True)


def merge_profile(
    z: Floats, root: int, merges: int = PROFILE_MERGES
) -> list[dict[str, float | int]]:
    """The last ``merges`` merges of one cluster: height and the sizes of the two children."""
    n = len(z) + 1
    out = []
    for row in subtree_merges(z, root)[:merges]:
        a, b = node_size(z, int(z[row, 0])), node_size(z, int(z[row, 1]))
        small, large = min(a, b), max(a, b)
        out.append(
            {
                'height': float(z[row, 2]),
                'small_child': small,
                'large_child': large,
                'small_share_of_large': small / large,
                'node': n + row,
            }
        )
    return out


def profile_summary(profile: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Counts that separate a run of small additions from a split into two large children."""
    additions = sum(p['small_share_of_large'] < SMALL_CHILD_SHARE for p in profile)
    return {
        'merges_listed': len(profile),
        'additions_under_5pct_of_sibling': additions,
        'crops_added_by_additions': sum(
            p['small_child'] for p in profile if p['small_share_of_large'] < SMALL_CHILD_SHARE
        ),
        'height_span': float(profile[0]['height'] - profile[-1]['height']) if profile else 0.0,
        'top_merge_small_share': float(profile[0]['small_share_of_large']) if profile else 0.0,
    }


def chaining_report(z: Floats, cut: float) -> dict[str, Any]:
    """Cluster count, size shares and the merge profile of the largest cluster at one cut."""
    n = len(z) + 1
    roots = cluster_roots(z, cut)
    sizes = np.array([node_size(z, r) for r in roots])
    biggest = roots[int(np.argmax(sizes))]
    profile = merge_profile(z, biggest)
    return {
        'cut': cut,
        'key_crops': n,
        'clusters': len(roots),
        'largest_cluster_crops': int(sizes.max()),
        'largest_cluster_share': float(sizes.max() / n),
        'share_in_clusters_of_3_or_more': float(sizes[sizes >= MIN_GROUP].sum() / n),
        'largest_cluster_profile': profile,
        'largest_cluster_profile_summary': profile_summary(profile),
    }
