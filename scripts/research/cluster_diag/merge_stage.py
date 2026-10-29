"""Judge-gated merge inside one key (decision note section 0.3), simulated with ground truth.

Start from the key's average-linkage partition at ``cut``. Walk the dendrogram nodes whose height
lies in ``(cut, cut + window]`` from the lowest up. A node is checkable when both its children are
current units (a cut cluster, or a node already accepted). The two units go to a judge on a pair
sheet ("same kind of object?"); a "same" merges them, anything else leaves them apart and blocks
every ancestor. Each check costs one decision.

The ground truth of a pair is whether the two units' plurality classes (over the pool's labelled
members) are the same class. The judge is a coin with two measured rates: it says "same" for a truly
same pair with ``p_same`` and for a truly different pair with ``p_false_same``. A unit with no labelled
pool member cannot be judged by the simulation, so its node is not checked and blocks its ancestors.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.cluster.hierarchy import fcluster


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    Ints = NDArray[np.integer[Any]]

# Haiku on COCO crops, pair sheets (the judge agreement run E0, 144 pair rows): false-"same" 0.030; accuracy 0.910 on decided rows with half the pairs the
# same, so P(same | truly same, decided) = 2 * 0.910 - 0.970 = 0.850; 6.9 % of rows are unsure and
# count as "not same": 0.850 * (1 - 0.069) = 0.791.
P_FALSE_SAME = 0.030
P_SAME = 0.791


@dataclass(frozen=True)
class Judge:
    p_same: float = P_SAME
    p_false_same: float = P_FALSE_SAME

    def says_same(self, truly_same: bool, rng: np.random.Generator) -> bool:
        return bool(rng.random() < (self.p_same if truly_same else self.p_false_same))


@dataclass(frozen=True)
class MergeResult:
    codes: Ints
    """Unit code of every leaf of the key."""
    checks: int
    accepted: int
    true_same_accepted: int
    false_same_accepted: int
    """Accepted merges whose two units had different plurality classes."""


def _plurality(c: Counter[str]) -> str:
    return min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def merge_key(
    z: Any,
    labels: Sequence[str | None],
    cut: float,
    window: float,
    judge: Judge,
    seed: int,
) -> MergeResult:
    """Merge inside one key. ``z`` is its linkage matrix, ``labels`` the pool label of each leaf
    (``None`` for a leaf outside the pool or unlabelled)."""
    n = len(labels)
    base = np.asarray(fcluster(z, cut, 'distance'), dtype=np.int64) - 1
    parent = list(range(int(base.max()) + 1))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    counts: dict[int, Counter[str]] = {}
    for leaf, lab in enumerate(labels):
        if lab is not None:
            counts.setdefault(int(base[leaf]), Counter())[lab] += 1
    rep = list(range(n))  # a leaf inside each node, to find the node's cut cluster
    for i in range(len(z)):
        rep.append(rep[int(z[i, 0])])
    accepted_node = [False] * (n + len(z))
    checks = accepted = true_same = false_same = 0
    for i in range(len(z)):
        height = float(z[i, 2])
        if height <= cut:
            continue
        if height > cut + window:
            break
        kids = (int(z[i, 0]), int(z[i, 1]))
        if not all(k < n or float(z[k - n, 2]) <= cut or accepted_node[k] for k in kids):
            continue
        a, b = (find(int(base[rep[k]])) for k in kids)
        if a == b or a not in counts or b not in counts:
            continue
        checks += 1
        truly_same = _plurality(counts[a]) == _plurality(counts[b])
        rng = np.random.default_rng([seed, i])
        if not judge.says_same(truly_same, rng):
            continue
        accepted += 1
        true_same += truly_same
        false_same += not truly_same
        parent[b] = a
        counts[a] = counts[a] + counts.pop(b)
        accepted_node[n + i] = True
    codes = np.array([find(int(base[leaf])) for leaf in range(n)], dtype=np.int64)
    return MergeResult(codes, checks, accepted, true_same, false_same)


def merged_partition(
    linkages: Sequence[tuple[Ints, Any]],
    n_total: int,
    labels: Sequence[str | None],
    cut: float,
    window: float,
    judge: Judge,
    seed: int,
) -> tuple[Ints, dict[str, int]]:
    """``merge_key`` over every key. ``linkages`` are ``partitions.group_linkages`` pairs (row
    indices of a key, its linkage or ``None``); ``labels`` hold one pool label per row of the whole
    set. Returns one unit code per row (unique across keys) and the totals."""
    codes = np.zeros(n_total, dtype=np.int64)
    offset = 0
    totals = {'checks': 0, 'accepted': 0, 'true_same_accepted': 0, 'false_same_accepted': 0}
    for g, (idx, z) in enumerate(linkages):
        part: Ints
        if z is None:
            part = np.zeros(len(idx), dtype=np.int64)
        else:
            res = merge_key(z, [labels[int(r)] for r in idx], cut, window, judge, seed * 1000 + g)
            part = res.codes
            totals['checks'] += res.checks
            totals['accepted'] += res.accepted
            totals['true_same_accepted'] += res.true_same_accepted
            totals['false_same_accepted'] += res.false_same_accepted
        dense = np.unique(part, return_inverse=True)[1].astype(np.int64)
        codes[idx] = dense + offset
        offset += int(dense.max()) + 1
    return codes, totals
