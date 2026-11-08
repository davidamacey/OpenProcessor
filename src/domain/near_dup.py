"""Near-duplicate rules: the closed lists of the ``near_dup_*`` tables, the defaults, the grouping.

The ``CHECK`` constraints are built from the tuples in ``src/db/models/near_dup.py``. The grouping
is deterministic: the same edges, decisions and vectors give the same groups whatever the order
they arrive in. It follows the research plan: connected components over the edges at or above a
cut, human decisions applied first (``different`` removes the edge, ``same`` adds one), a component
larger than the size guard split by complete linkage, the representative the most central member
(the member id breaks a tie), and lineage carried from the parent run by member Jaccard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence


LEVELS: tuple[str, ...] = ('image', 'item', 'core')
PROJECT_LEVELS: tuple[str, ...] = ('image', 'item')
PURPOSES: tuple[str, ...] = ('dedup', 'leak')
PENDING_REASONS: tuple[str, ...] = (
    'added',
    'vector_changed',
    'boxes_changed',
    'deleted',
    'group_changed',
)
RUN_MODES: tuple[str, ...] = ('full', 'incremental')
RUN_STATUSES: tuple[str, ...] = ('building', 'current', 'superseded', 'failed')
EDGE_KINDS: tuple[str, ...] = ('embedding',)
DECISIONS: tuple[str, ...] = ('same', 'different')

# Today's behaviour (0.98 cosine on the whole frame and on the crop) until the prototype runs
# say otherwise.
ALGORITHM_VERSION = 'nd-1'
DEFAULT_IMAGE_DEDUP_COSINE = 0.98
DEFAULT_ITEM_DEDUP_COSINE = 0.98
DEFAULT_K = 64
DEFAULT_MAX_GROUP_SIZE = 1000
MIN_K = 8
MAX_K = 256
MIN_LEAK_COSINE = 0.5
MIN_GROUP_SIZE_GUARD = 2
LINEAGE_MIN_JACCARD = 0.5
# Runs kept per (project, level) besides the current one.
KEPT_SUPERSEDED_RUNS = 2


def order_pair(a: str, b: str) -> tuple[str, str]:
    """The stored order of a pair: ``a_id < b_id``."""
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class Group:
    """One group of a run: members sorted by id, the representative one of them."""

    members: tuple[str, ...]
    representative: str
    cosine_to_rep: Mapping[str, float]
    min_edge_cosine: float
    min_cosine_to_rep: float


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self._parent
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # The smaller id is the root, so the result never depends on the edge order.
            low, high = (ra, rb) if ra < rb else (rb, ra)
            self._parent[high] = low


def effective_edges(
    edges: Iterable[tuple[str, str, float]],
    cut: float,
    decisions: Mapping[tuple[str, str], str] | None = None,
) -> dict[tuple[str, str], float]:
    """Edges at or above ``cut`` with the human decisions applied.

    A ``different`` decision removes its edge; a ``same`` decision adds one (score 1.0 when no
    edge exists, so it also survives the size-guard split).
    """
    kept: dict[tuple[str, str], float] = {}
    for a, b, score in edges:
        if a != b and score >= cut:
            pair = order_pair(a, b)
            kept[pair] = max(score, kept.get(pair, score))
    for pair, decision in (decisions or {}).items():
        ordered = order_pair(*pair)
        if decision == 'different':
            kept.pop(ordered, None)
        elif decision == 'same':
            kept[ordered] = max(kept.get(ordered, 0.0), 1.0)
    return kept


def connected_components(edges: Mapping[tuple[str, str], float]) -> list[list[str]]:
    """Components of size >= 2, members sorted, components sorted by their first member."""
    nodes = {n for pair in edges for n in pair}
    forest = _UnionFind(nodes)
    for a, b in sorted(edges):
        forest.union(a, b)
    by_root: dict[str, list[str]] = {}
    for node in sorted(nodes):
        by_root.setdefault(forest.find(node), []).append(node)
    return sorted(by_root.values(), key=lambda c: c[0])


def split_complete_linkage(
    component: Sequence[str], edges: Mapping[tuple[str, str], float], max_group_size: int
) -> list[list[str]]:
    """Split one component into groups no larger than ``max_group_size``.

    Greedy agglomeration over the component's edges, strongest first: two clusters merge only
    when every cross pair has an edge (complete linkage) and the merged size stays within the
    guard. Edge order is (score descending, ids ascending), so the split is deterministic.
    """
    members = set(component)
    inside = {p: s for p, s in edges.items() if p[0] in members and p[1] in members}
    clusters: dict[str, list[str]] = {n: [n] for n in component}
    owner = {n: n for n in component}
    for (a, b), _score in sorted(inside.items(), key=lambda kv: (-kv[1], kv[0])):
        ca, cb = owner[a], owner[b]
        if ca == cb:
            continue
        if len(clusters[ca]) + len(clusters[cb]) > max_group_size:
            continue
        if not all(order_pair(x, y) in inside for x in clusters[ca] for y in clusters[cb]):
            continue
        low, high = (ca, cb) if ca < cb else (cb, ca)
        clusters[low].extend(clusters.pop(high))
        for member in clusters[low]:
            owner[member] = low
    return sorted((sorted(c) for c in clusters.values() if len(c) >= 2), key=lambda c: c[0])


@dataclass(frozen=True)
class Components:
    """Member sets of one purpose and how many components the size guard had to split."""

    groups: list[list[str]]
    oversized_split: int


def group_members(edges: Mapping[tuple[str, str], float], max_group_size: int) -> Components:
    """Components with the size guard applied (``edges`` already at the cut, decisions applied)."""
    groups: list[list[str]] = []
    split = 0
    for component in connected_components(edges):
        if len(component) <= max_group_size:
            groups.append(component)
            continue
        split += 1
        groups.extend(split_complete_linkage(component, edges, max_group_size))
    return Components(sorted(groups, key=lambda g: g[0]), split)


def most_central(mat: np.ndarray, group: Sequence[int]) -> int:
    """Index (into ``group``) of the most central member: the largest summed cosine.

    ``np.argmax`` returns the first maximum, so with rows ordered by member id a tie goes to the
    smaller id.
    """
    sub = mat[list(group)]
    centrality = (sub @ sub.T).sum(axis=1)
    return int(np.argmax(centrality))


@dataclass(frozen=True)
class Seat:
    """Where one member sits in the current run's dedup groups."""

    group_id: str
    is_representative: bool
    cosine_to_rep: float


def collapse_drops(seats: Mapping[str, Seat], preferred: Collection[str] = ()) -> set[str]:
    """The members to drop so each dedup group keeps one survivor among ``seats``.

    A preferred member (a held-out frame) wins, so its near-twins leave the training pool rather
    than the held-out frame. Otherwise the stored representative wins; when the export holds only
    other members of the group, the one closest to the representative. The member id breaks ties.
    """
    groups: dict[str, list[str]] = {}
    for member, seat in seats.items():
        groups.setdefault(seat.group_id, []).append(member)
    dropped: set[str] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        pool = [m for m in members if m in preferred] or members
        winner = min(
            pool, key=lambda m: (not seats[m].is_representative, -seats[m].cosine_to_rep, m)
        )
        dropped.update(m for m in members if m != winner)
    return dropped


def describe_group(
    members: Sequence[str],
    vectors: np.ndarray,
    edges: Mapping[tuple[str, str], float],
) -> Group:
    """Representative and cosines of one group; ``vectors`` rows follow ``members`` (sorted)."""
    ordered = list(members)
    rep_index = most_central(vectors, range(len(ordered)))
    cosines = vectors @ vectors[rep_index]
    to_rep = {m: float(cosines[i]) for i, m in enumerate(ordered)}
    inner = [s for (a, b), s in edges.items() if a in to_rep and b in to_rep]
    return Group(
        members=tuple(ordered),
        representative=ordered[rep_index],
        cosine_to_rep=to_rep,
        min_edge_cosine=min(inner) if inner else 1.0,
        min_cosine_to_rep=min(c for m, c in to_rep.items() if m != ordered[rep_index]),
    )


def assign_lineage(
    new_groups: Mapping[str, Iterable[str]],
    parent_groups: Mapping[str, tuple[str, Iterable[str]]],
) -> dict[str, str | None]:
    """Carry a parent group's lineage onto the new group it overlaps most (Jaccard >= 0.5).

    ``new_groups`` maps a new group id to its members; ``parent_groups`` maps a parent group id
    to ``(lineage_id, members)``. A parent lineage goes to at most one new group; the answer
    maps each new group id to the carried lineage id, or ``None`` when it starts its own.
    """
    new_sets = {gid: frozenset(m) for gid, m in new_groups.items()}
    parent_sets = {gid: (lin, frozenset(m)) for gid, (lin, m) in parent_groups.items()}
    candidates: list[tuple[float, str, str]] = []
    for new_id, new_members in new_sets.items():
        for parent_id, (_lineage, parent_members) in parent_sets.items():
            overlap = len(new_members & parent_members)
            if not overlap:
                continue
            jaccard = overlap / len(new_members | parent_members)
            if jaccard >= LINEAGE_MIN_JACCARD:
                candidates.append((jaccard, new_id, parent_id))
    carried: dict[str, str | None] = dict.fromkeys(new_sets)
    used: set[str] = set()
    for _jaccard, new_id, parent_id in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
        lineage = parent_sets[parent_id][0]
        if carried[new_id] is None and lineage not in used:
            carried[new_id] = lineage
            used.add(lineage)
    return carried
