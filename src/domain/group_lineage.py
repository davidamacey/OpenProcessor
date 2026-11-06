"""Identity carry-over between two groupings of the same items (clustering and near-duplicates).

A new grouping keeps the identity of an old group when their members overlap enough: the
Jaccard matrix of old against new groups over shared members, then an optimal one-to-one
assignment (Hungarian algorithm, ``scipy.optimize.linear_sum_assignment``) over the pairs at or
above ``min_jaccard``. Everything else is classified by how the old groups' members were
redistributed (the MONIC transition taxonomy: survival, split, merge, absorption, new).

Pure: it sees only keys and member collections, no storage. The overlap graph is split into
connected components first, so the dense assignment problem is solved per component and the cost
follows the overlap structure, not ``old x new`` (an 860,000-item grouping has thousands of
groups). References:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
and Spiliopoulou et al., "MONIC: modeling and monitoring cluster transitions", KDD 2006.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

Transition = Literal['new', 'survived', 'split', 'merged', 'absorbed']

MIN_JACCARD = 0.5
SPLIT_SHARE = 0.5
"""A new group is a split part when one old group supplies at least this share of its members."""
MERGE_SHARE = 0.25
"""A new group is a merge when two or more old groups each supply at least this share."""
ABSORBED_SHARE = 0.5
"""An unmatched old group is absorbed when this share of its members sit in one matched new group."""


@dataclass(frozen=True)
class GroupLineage:
    """What became of one new group."""

    transition: Transition
    predecessors: tuple[Any, ...] = ()
    """Old group keys the new group descends from (the matched one first), sorted by share."""
    jaccard: float | None = None
    """Jaccard against the matched old group; ``None`` unless ``transition`` is survived/absorbed."""

    @property
    def inherited(self) -> Any | None:
        """The old key whose identity the new group keeps: only a match inherits."""
        if self.transition in ('survived', 'absorbed') and self.predecessors:
            return self.predecessors[0]
        return None


@dataclass(frozen=True)
class LineageResult:
    by_new: dict[Any, GroupLineage] = field(default_factory=dict)
    dropped: tuple[Any, ...] = ()
    """Old groups that no new group descends from (they ended)."""


def _pair_overlaps(
    old: Mapping[Any, Collection[str]], new: Mapping[Any, Collection[str]]
) -> Counter[tuple[Any, Any]]:
    where: dict[str, Any] = {}
    for key, members in old.items():
        for member in members:
            where[member] = key
    overlaps: Counter[tuple[Any, Any]] = Counter()
    for new_key, members in new.items():
        for member in members:
            old_key = where.get(member)
            if old_key is not None:
                overlaps[(old_key, new_key)] += 1
    return overlaps


def _assign(
    pairs: dict[tuple[Any, Any], float],
    old_keys: list[Any],
    new_keys: list[Any],
) -> dict[Any, tuple[Any, float]]:
    """Optimal one-to-one matching of ``new -> (old, jaccard)`` over the pairs given."""
    if not pairs:
        return {}
    old_ix = {key: i for i, key in enumerate(old_keys)}
    new_ix = {key: i for i, key in enumerate(new_keys)}
    rows = np.fromiter((old_ix[o] for o, _ in pairs), dtype=np.int64, count=len(pairs))
    cols = np.fromiter((new_ix[n] for _, n in pairs), dtype=np.int64, count=len(pairs))
    n_old, n_new = len(old_keys), len(new_keys)
    graph = coo_matrix((np.ones(len(pairs)), (rows, n_old + cols)), shape=(n_old + n_new,) * 2)
    _, component = connected_components(graph, directed=False)
    by_component: dict[int, list[int]] = defaultdict(list)
    for idx, (o, _n) in enumerate(pairs):
        by_component[int(component[old_ix[o]])].append(idx)
    pair_list = list(pairs.items())
    matched: dict[Any, tuple[Any, float]] = {}
    for indices in by_component.values():
        olds = sorted({pair_list[i][0][0] for i in indices}, key=old_ix.__getitem__)
        news = sorted({pair_list[i][0][1] for i in indices}, key=new_ix.__getitem__)
        o_local = {key: i for i, key in enumerate(olds)}
        n_local = {key: i for i, key in enumerate(news)}
        cost = np.zeros((len(olds), len(news)))
        for i in indices:
            (o, n), score = pair_list[i]
            cost[o_local[o], n_local[n]] = -score
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind, strict=True):
            score = -float(cost[r, c])
            if score > 0.0:
                matched[news[c]] = (olds[r], score)
    return matched


def match_groups(
    old: Mapping[Any, Collection[str]],
    new: Mapping[Any, Collection[str]],
    min_jaccard: float = MIN_JACCARD,
) -> LineageResult:
    """Match ``new`` groups to ``old`` groups by member overlap and classify every new group.

    Deterministic for equal inputs: ties in the assignment break by key order in the mappings.
    Equal members under a new grouping always keep their identity (Jaccard 1.0), however the
    groups are numbered or ordered.
    """
    old_sizes = {key: len(set(members)) for key, members in old.items()}
    new_sizes = {key: len(set(members)) for key, members in new.items()}
    overlaps = _pair_overlaps(
        {k: set(m) for k, m in old.items()}, {k: set(m) for k, m in new.items()}
    )
    jaccard = {
        pair: count / (old_sizes[pair[0]] + new_sizes[pair[1]] - count)
        for pair, count in overlaps.items()
    }
    eligible = {pair: j for pair, j in jaccard.items() if j >= min_jaccard}
    matched = _assign(eligible, list(old), list(new))
    matched_old = {old_key for old_key, _ in matched.values()}

    by_new: dict[Any, GroupLineage] = {}
    for new_key in new:
        if new_key in matched:
            old_key, score = matched[new_key]
            absorbed = tuple(
                sorted(
                    (
                        o
                        for (o, n), c in overlaps.items()
                        if n == new_key
                        and o not in matched_old
                        and c / old_sizes[o] >= ABSORBED_SHARE
                    ),
                    key=lambda o: -overlaps[(o, new_key)],
                )
            )
            by_new[new_key] = GroupLineage(
                'absorbed' if absorbed else 'survived', (old_key, *absorbed), score
            )
            continue
        shares = sorted(
            (
                (o, c / new_sizes[new_key])
                for (o, n), c in overlaps.items()
                if n == new_key and new_sizes[new_key]
            ),
            key=lambda pair: -pair[1],
        )
        merged = [o for o, share in shares if share >= MERGE_SHARE]
        if len(merged) >= 2:
            by_new[new_key] = GroupLineage('merged', tuple(merged))
        elif shares and shares[0][1] >= SPLIT_SHARE:
            by_new[new_key] = GroupLineage('split', (shares[0][0],))
        else:
            by_new[new_key] = GroupLineage('new')
    descended = {p for lineage in by_new.values() for p in lineage.predecessors}
    dropped = tuple(key for key in old if key not in descended)
    return LineageResult(by_new, dropped)


__all__ = [
    'ABSORBED_SHARE',
    'MERGE_SHARE',
    'MIN_JACCARD',
    'SPLIT_SHARE',
    'GroupLineage',
    'LineageResult',
    'Transition',
    'match_groups',
]
