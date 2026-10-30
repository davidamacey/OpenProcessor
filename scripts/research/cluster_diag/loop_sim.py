"""The auto-label loop's recursion on root clusters, with an optional hold-back guard (Z4).

``gate_sim`` simulates one look at each cluster. The loop also splits a cluster that fails the gate
and tries the children, reusing the answers the parent already has. This module follows
``_Node.run``, ``_decide_kind`` and ``_split_children`` of the loop (branch
``feat/v070-assist-loop``, commit ``ba9e27ad``, files ``assist_loop/evaluate.py``, ``node_step.py``
and ``splitter.py``):

* a node's first sample is its reused answers (every answer an ancestor got for one of its members)
  plus a draw up to ``required_sample(n, theta, alpha / 2, 0)`` in total; one to ``max_dissent``
  dissents with too few answers top the sample up once to ``required_sample(..., dissent)``;
* a node that passes labels its unasked members with the top class; one that fails goes to review
  at depth ``MAX_DEPTH`` or under ``MIN_NODE`` members, else it is split into ``BRANCHING`` children
  by complete-linkage cosine AHC (seeded k-means above ``MAX_REFINE`` members) and each child is a
  node of its own.

What differs from the loop: a member is a pool index, not a crop id, so the draws are seeded by
``(seed, path)`` but are not the loop's ``random.Random`` draws; there is no budget and no failed
answer. The labeler is an oracle (optionally wrong with probability ``noise``).

Guard (label-free): before a passing node propagates, any unasked member whose cosine distance to
the sample centroid exceeds the 95th percentile of the sampled members' own distances is held back
and goes to review instead.

A decision is one labeler answer or one review. Every item is decided at most once: it is asked
(sampled in some node), reviewed (a review leaf, or held back), or labelled for free.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.research.cluster_diag import gate_sim as gs


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from numpy.typing import NDArray

    Ints = NDArray[np.integer[Any]]
    Floats = NDArray[np.floating[Any]]
    Splitter = Callable[[Ints, str], list[Ints]]

BRANCHING = 4
MAX_DEPTH = 3
MIN_NODE = 40
MAX_REFINE = 8000  # above this the loop switches from AHC to k-means
MIN_REFINE = 4
KMEANS_N_INIT = 4
GUARD_PERCENTILE = 95.0


def path_seed(seed: int, path: str) -> int:
    """A stable 32-bit integer for ``(seed, path)`` (``loop_tree.seed_int``)."""
    return int(hashlib.sha256(f'{seed}:{path}'.encode()).hexdigest()[:8], 16)


class AhcSplitter:
    """The loop's ``split_node`` on pool vectors: children sorted by their smallest index.

    AHC splits do not depend on the sampling seed, so they are cached by membership across seeds.
    """

    def __init__(self, vectors: Floats, seed: int = 0) -> None:
        self.vectors = vectors
        self.seed = seed
        self._cache: dict[bytes, list[Ints]] = {}

    def __call__(self, members: Ints, path: str) -> list[Ints]:
        n = len(members)
        if n < max(MIN_REFINE, BRANCHING):
            return []
        if n > MAX_REFINE:
            return self._kmeans(members, path)
        key = members.tobytes()
        if key not in self._cache:
            self._cache[key] = self._ahc(members)
        return self._cache[key]

    def _ahc(self, members: Ints) -> list[Ints]:
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(
            n_clusters=BRANCHING, linkage='complete', metric='cosine'
        ).fit_predict(self.vectors[members])
        return _children(members, labels)

    def _kmeans(self, members: Ints, path: str) -> list[Ints]:
        from sklearn.cluster import KMeans

        unit = self.vectors[members]
        distinct = len(np.unique(unit, axis=0))
        if distinct < 2:
            return []
        labels = KMeans(
            n_clusters=min(BRANCHING, distinct),
            random_state=path_seed(self.seed, path),
            n_init=KMEANS_N_INIT,
        ).fit_predict(unit)
        return _children(members, labels)


def _children(members: Ints, labels: Ints) -> list[Ints]:
    groups = [np.sort(members[labels == c]) for c in np.unique(labels)]
    kids = sorted((g for g in groups if len(g)), key=lambda g: int(g[0]))
    return kids if len(kids) >= 2 else []


@dataclass(frozen=True)
class Look:
    top: str
    agree: int
    answered: int
    need: int
    passed: bool
    may_top_up: bool


def judge(answers: Sequence[str], size: int, registry: frozenset[str] | None) -> Look:
    """``evaluate.judge``: the plurality class (ties to the smaller name) and the gate on it.

    A top class outside ``registry`` (when given) cannot pass and gets no top-up.
    """
    votes: dict[str, int] = {}
    for a in answers:
        votes[a] = votes.get(a, 0) + 1
    top, agree = min(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    answered = len(answers)
    dissent = answered - agree
    need = gs.required_sample(size, gs.THETA, gs.ALPHA_STAGE, dissent)
    eligible_class = registry is None or top in registry
    within = dissent <= gs.MAX_DISSENT
    enough = answered >= need
    return Look(
        top,
        agree,
        answered,
        need,
        eligible_class and within and enough,
        eligible_class and within and not enough,
    )


def held_back(vectors: Floats, members: Ints, sampled: Ints, percentile: float) -> Ints:
    """Members of ``members`` not in ``sampled`` beyond the percentile of the sample's own cosine
    distance to the sample centroid. Vectors are unit length."""
    centroid = vectors[sampled].mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return np.empty(0, dtype=np.int64)
    centroid = centroid / norm
    limit = float(np.percentile(1.0 - vectors[sampled] @ centroid, percentile))
    rest = np.setdiff1d(members, sampled)
    return np.asarray(rest[1.0 - vectors[rest] @ centroid > limit], dtype=np.int64)


@dataclass
class Run:
    """Everything one seed of one arm decided."""

    asked: dict[int, str] = field(default_factory=dict)
    review: set[int] = field(default_factory=set)
    propagated: dict[int, str] = field(default_factory=dict)
    held: dict[int, str] = field(default_factory=dict)
    """Held-back index -> the class it would have got."""
    nodes: dict[str, int] = field(default_factory=dict)


def _bump(run: Run, kind: str) -> None:
    run.nodes[kind] = run.nodes.get(kind, 0) + 1


class _Loop:
    """One seed of the loop over the roots of a partition."""

    def __init__(
        self,
        truth: Sequence[str],
        vectors: Floats,
        splitter: Splitter,
        *,
        seed: int,
        guard: bool,
        noise: float,
        registry: frozenset[str] | None,
    ) -> None:
        self.t = np.asarray(truth).astype(str)
        self.classes = sorted(set(self.t.tolist()))
        self.vectors, self.splitter, self.seed = vectors, splitter, seed
        self.guard, self.noise, self.registry = guard, noise, registry
        self.run = Run()

    def answer(self, i: int) -> str:
        if self.noise > 0:
            rng = np.random.default_rng([self.seed, 7, i])
            if rng.random() < self.noise:
                others = [c for c in self.classes if c != self.t[i]]
                return others[int(rng.integers(len(others)))] if others else str(self.t[i])
        return str(self.t[i])

    def ask(self, ids: Iterable[int]) -> None:
        for i in ids:
            self.run.asked[int(i)] = self.answer(int(i))

    def draw(self, members: Ints, count: int, path: str, tag: str) -> Ints:
        free = np.asarray([i for i in members.tolist() if i not in self.run.asked], dtype=np.int64)
        count = min(max(count, 0), len(free))
        if count == 0:
            return np.empty(0, dtype=np.int64)
        rng = np.random.default_rng([self.seed, path_seed(self.seed, path + tag)])
        return np.sort(rng.choice(free, size=count, replace=False))

    def sampled_of(self, members: Ints) -> Ints:
        return np.asarray([i for i in members.tolist() if i in self.run.asked], dtype=np.int64)

    def look(self, members: Ints) -> Look:
        answers = [self.run.asked[i] for i in self.sampled_of(members).tolist()]
        return judge(answers, len(members), self.registry)

    def evaluate(self, members: Ints, path: str) -> Look:
        """``_Node.run``: reused answers plus a draw to the no-dissent size, then one top-up."""
        first = max(gs.required_sample(len(members), gs.THETA, gs.ALPHA_STAGE, 0), 1)
        self.ask(self.draw(members, first - len(self.sampled_of(members)), path, '#1'))
        verdict = self.look(members)
        if verdict.may_top_up:
            self.ask(self.draw(members, verdict.need - verdict.answered, path, '#topup'))
            verdict = self.look(members)
        return verdict

    def propagate(self, members: Ints, top: str) -> None:
        sample = self.sampled_of(members)
        rest = np.setdiff1d(members, sample)
        held = (
            held_back(self.vectors, members, sample, GUARD_PERCENTILE) if self.guard else rest[:0]
        )
        for i in held.tolist():
            self.run.held[i] = top
            self.run.review.add(i)
        for i in np.setdiff1d(rest, held).tolist():
            self.run.propagated[i] = top

    def node(self, members: Ints, path: str, level: int) -> None:
        verdict = self.evaluate(members, path)
        if verdict.passed:
            _bump(self.run, 'pass')
            self.propagate(members, verdict.top)
            return
        unasked = np.setdiff1d(members, self.sampled_of(members))
        kids: list[Ints] = []
        if level >= MAX_DEPTH:
            kind = 'review_leaf'
        elif len(members) < MIN_NODE:
            kind = 'review_small'
        else:
            kids = self.splitter(members, path)
            kind = 'review_leaf'  # a node that cannot be split into two children
        if not kids:
            _bump(self.run, kind)
            self.run.review.update(unasked.tolist())
            return
        _bump(self.run, 'split')
        for index, kid in enumerate(kids):
            self.node(kid, f'{path}.{index}', level + 1)


def run_loop(
    codes: Ints,
    truth: Sequence[str],
    vectors: Floats,
    splitter: Splitter,
    *,
    seed: int,
    guard: bool = False,
    noise: float = 0.0,
    registry: frozenset[str] | None = None,
) -> Run:
    """The loop over every root cluster of ``codes`` (one code per pool item)."""
    loop = _Loop(truth, vectors, splitter, seed=seed, guard=guard, noise=noise, registry=registry)
    codes = np.asarray(codes)
    for code in sorted(set(codes.tolist())):
        loop.node(np.nonzero(codes == code)[0].astype(np.int64), f'r{code}', 0)
    return loop.run


def measure(run: Run, truth: Sequence[str], open_set: Iterable[str] = ()) -> dict[str, Any]:
    """U1 to U3, the open-set figures and what the guard held back, for one run."""
    t = np.asarray(truth).astype(str)
    n = len(t)
    opened = set(open_set)
    decided = set(run.asked) | run.review
    if set(run.asked) & run.review or set(run.propagated) & decided:
        msg = 'an item was decided twice (asked, reviewed or propagated)'
        raise ValueError(msg)
    hit = {i: bool(t[i] == c) for i, c in run.propagated.items()}
    open_ids = [i for i in run.propagated if t[i] in opened]
    held_wrong = sum(1 for i, c in run.held.items() if t[i] != c)
    return {
        'items': n,
        'decisions': len(decided),
        'u1': len(decided) / n if n else math.nan,
        'propagated': len(run.propagated),
        'u2': sum(hit.values()) / len(hit) if hit else math.nan,
        'u3': len(run.propagated) / n if n else math.nan,
        'wrong_propagated': len(hit) - sum(hit.values()),
        'open_propagated': len(open_ids),
        'open_correct': sum(hit[i] for i in open_ids),
        'open_wrong': sum(1 for i in open_ids if not hit[i]),
        'u2_open': sum(hit[i] for i in open_ids) / len(open_ids) if open_ids else math.nan,
        'asked': len(run.asked),
        'reviewed': len(run.review),
        'held_back': len(run.held),
        'held_back_would_be_wrong': held_wrong,
        'held_back_open_set': sum(1 for i in run.held if t[i] in opened),
        'nodes': dict(run.nodes),
    }


METRICS = (
    'u1',
    'u2',
    'u3',
    'u2_open',
    'propagated',
    'wrong_propagated',
    'open_propagated',
    'open_wrong',
    'asked',
    'reviewed',
    'held_back',
    'held_back_would_be_wrong',
    'held_back_open_set',
)


def seeded_loop(
    codes: Ints,
    truth: Sequence[str],
    vectors: Floats,
    splitter: Splitter,
    *,
    seeds: Sequence[int],
    open_set: Iterable[str] = (),
    guard: bool = False,
    noise: float = 0.0,
    registry: frozenset[str] | None = None,
) -> dict[str, Any]:
    """``run_loop`` at every seed; each metric as mean, min and max over the seeds that define it,
    plus the per-seed values (the paired comparison between arms needs them)."""
    opened = frozenset(open_set)
    runs = [
        measure(
            run_loop(
                codes,
                truth,
                vectors,
                splitter,
                seed=s,
                guard=guard,
                noise=noise,
                registry=registry,
            ),
            truth,
            opened,
        )
        for s in seeds
    ]
    out: dict[str, Any] = {
        'seeds': list(seeds),
        'noise': noise,
        'guard': guard,
        'items': runs[0]['items'],
        'clusters': len(set(np.asarray(codes).tolist())),
        'per_seed': {k: [r[k] for r in runs] for k in METRICS},
        'nodes_mean': {
            kind: float(np.mean([r['nodes'].get(kind, 0) for r in runs]))
            for kind in sorted({k for r in runs for k in r['nodes']})
        },
    }
    for key in METRICS:
        vals = [r[key] for r in runs if not (isinstance(r[key], float) and math.isnan(r[key]))]
        out[key] = (
            {'mean': float(np.mean(vals)), 'min': float(min(vals)), 'max': float(max(vals))}
            if vals
            else None
        )
    return out
