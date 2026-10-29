"""Split by descent (Y4): do the children of an impure cluster, one or two levels down the key's
dendrogram, separate the labels?

The question is about clusters a sampled gate would reject: those whose majority label covers less
than ``IMPURE_BELOW`` of their labelled members. Descending replaces a cluster by the two children
of its last merge (level 1) and each internal child by its own two children (level 2); leaves
stay. Purity of a descent is item-weighted: labelled members that carry their part's majority
label, over all labelled members. Only labelled crops of the fitting half are looked at.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.research.cluster_diag import chaining as ch


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    Floats = NDArray[np.floating[Any]]

IMPURE_BELOW = 0.9  # majority share under which a cluster counts as rejected
PURE_AT = 0.9  # item-weighted purity a descent has to reach
MIN_LABELLED = 5  # clusters with fewer labelled members are not judged (their share is noise)
MAX_LEVELS = 2
ADOPT_SHARE = 0.8  # fixed beforehand: adopt if at least this share of impure clusters reach PURE_AT


def leaves_of(z: Floats, node: int) -> list[int]:
    """Leaf ids under ``node`` (a leaf is its own single leaf)."""
    n = len(z) + 1
    out, stack = [], [node]
    while stack:
        cur = stack.pop()
        if cur < n:
            out.append(cur)
        else:
            stack.extend((int(z[cur - n, 0]), int(z[cur - n, 1])))
    return out


def descend(z: Floats, root: int, levels: int) -> list[int]:
    """The parts a cluster breaks into after ``levels`` levels of splitting (leaves stay)."""
    n = len(z) + 1
    parts = [root]
    for _ in range(levels):
        nxt: list[int] = []
        for node in parts:
            if node < n:
                nxt.append(node)
            else:
                nxt.extend((int(z[node - n, 0]), int(z[node - n, 1])))
        parts = nxt
    return parts


def majority_counts(leaves: Sequence[int], labels: Mapping[int, Any]) -> tuple[int, int]:
    """(labelled members, members carrying the most common label) of one set of leaves."""
    counts = Counter(labels[i] for i in leaves if i in labels)
    return sum(counts.values()), max(counts.values(), default=0)


def weighted_purity(parts: Sequence[Sequence[int]], labels: Mapping[int, Any]) -> float | None:
    """Item-weighted purity of a set of parts; ``None`` when no member is labelled."""
    total = kept = 0
    for leaves in parts:
        n, top = majority_counts(leaves, labels)
        total, kept = total + n, kept + top
    return kept / total if total else None


def judge_key(z: Floats, cut: float, labels: Mapping[int, Any]) -> list[dict[str, Any]]:
    """One row per judged cluster of a key at ``cut``: its majority share and, if impure, what
    each descent level gives. ``labels`` maps leaf id -> label for the fitting half only."""
    rows = []
    for root in ch.cluster_roots(z, cut):
        leaves = leaves_of(z, root)
        n, top = majority_counts(leaves, labels)
        if n < MIN_LABELLED:
            continue
        row: dict[str, Any] = {
            'labelled': n,
            'majority_share': top / n,
            'impure': top / n < IMPURE_BELOW,
        }
        if row['impure']:
            for level in range(1, MAX_LEVELS + 1):
                parts = [leaves_of(z, p) for p in descend(z, root, level)]
                row[f'purity_level_{level}'] = weighted_purity(parts, labels)
                row[f'parts_level_{level}'] = len(parts)
                row[f'labelled_in_parts_under_3_level_{level}'] = sum(
                    p for p in (majority_counts(x, labels)[0] for x in parts) if p < 3
                )
            row['reaches'] = any(
                (row[f'purity_level_{lv}'] or 0.0) >= PURE_AT for lv in range(1, MAX_LEVELS + 1)
            )
        rows.append(row)
    return rows


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts and the share of impure clusters that reach the purity bar within two levels."""
    impure = [r for r in rows if r['impure']]
    reach = [r for r in impure if r['reaches']]
    reach1 = [r for r in impure if (r['purity_level_1'] or 0.0) >= PURE_AT]
    share = len(reach) / len(impure) if impure else None
    return {
        'judged_clusters': len(rows),
        'impure_clusters': len(impure),
        'impure_labelled_members': sum(r['labelled'] for r in impure),
        'reach_within_two_levels': len(reach),
        'reach_at_level_1': len(reach1),
        'share_reaching': share,
        'median_parts_level_2': float(np.median([r['parts_level_2'] for r in impure]))
        if impure
        else None,
        'labelled_in_parts_under_3_level_2': sum(
            r['labelled_in_parts_under_3_level_2'] for r in impure
        ),
        'adopt_bar': ADOPT_SHARE,
        'verdict': 'INCONCLUSIVE'
        if share is None
        else ('PASS' if share >= ADOPT_SHARE else 'FAIL'),
    }
