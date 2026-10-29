"""The auto-label loop's propagation gate, simulated on a partition (decision note section 0.2).

The gate is the one on branch ``feat/v070-assist-loop`` (commit ``ba9e27ad``:
``src/domain/loop_stats.py`` and ``src/services/curation/assist_loop/evaluate.py``), with its
defaults: theta 0.9, alpha 0.05 (each look tested at alpha / 2), max dissent 1, and the two-stage
top-up. That branch is not on this one, so ``tail`` and ``required_sample`` below are copied from it
(a test holds them to the loop's own published tables), and ``run_gate`` follows ``judge`` and
``_Node.run`` step by step.

The labeler is an oracle (the ground-truth label is its answer) or, for the noisy row, an oracle
that is wrong independently with a given probability and then names another class at random. The
loop's own recursion into children is not simulated: a cluster that fails the gate has every member
labelled one by one.

Metrics over a pool of items (the labelled crops of one half):

* ``u1``: labeler decisions per item. A passing cluster costs its sample; a failing one all of it.
* ``u2``: share of propagated labels that equal the truth (items never asked, in passing clusters).
* ``u3``: share of items labelled by propagation.
* ``u2_open``: ``u2`` restricted to items whose true class is in ``open_set``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray

THETA = 0.9
ALPHA = 0.05
MAX_DISSENT = 1
ALPHA_STAGE = ALPHA / (MAX_DISSENT + 1)
NOISE = 0.10


def tail(n_members: int, agreeing: int, n: int, at_least: int) -> float:
    """``P(X >= at_least)`` for ``X`` agreeing members in a sample of ``n`` drawn without
    replacement from ``n_members`` of which ``agreeing`` agree (``loop_stats._tail``)."""
    total = math.comb(n_members, n)
    hit = sum(
        math.comb(agreeing, k) * math.comb(n_members - agreeing, n - k)
        for k in range(max(at_least, 0), min(n, agreeing) + 1)
    )
    return hit / total


@cache
def required_sample(n_members: int, theta: float, alpha: float, dissent: int) -> int:
    """Smallest sample for which ``n - dissent`` agreeing answers prove ``h >= theta`` at level
    ``alpha``; a node too small to ever prove it is asked whole (``loop_stats.required_sample``)."""
    below = math.ceil(theta * n_members) - 1
    for n in range(dissent + 1, n_members):
        if tail(n_members, below, n, n - dissent) <= alpha:
            return n
    return n_members


@dataclass(frozen=True)
class ClusterResult:
    size: int
    sampled: int
    passed: bool
    top: str
    propagated: tuple[int, ...]
    """Positions (in the cluster's member list) of the members labelled ``top`` without being
    asked; empty when the cluster failed."""


def _look(answers: Sequence[str], size: int) -> tuple[str, int, int, int, bool, bool]:
    """(top class, agree, answered, need, passed, may_top_up) of ``answers`` in a node of ``size``.

    The top class is the plurality, ties to the smaller name; ``judge`` takes the first of the
    proposal's distribution, which has no stated tie rule.
    """
    votes = Counter(answers)
    top, agree = min(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    answered = len(answers)
    dissent = answered - agree
    need = required_sample(size, THETA, ALPHA_STAGE, dissent)
    within = dissent <= MAX_DISSENT
    enough = answered >= need
    return top, agree, answered, need, within and enough, within and not enough


def run_gate(
    truth: Sequence[str],
    rng: np.random.Generator,
    *,
    noise: float = 0.0,
    classes: Sequence[str] = (),
) -> ClusterResult:
    """One cluster through the gate. ``truth`` are its members' true classes.

    Draw order is one seeded permutation: the first stage takes its head, the top-up the next
    members, which is a draw without replacement in two steps as the loop does it.
    """
    size = len(truth)
    order = rng.permutation(size)

    def answer(i: int) -> str:
        if noise > 0 and rng.random() < noise:
            others = [c for c in classes if c != truth[i]]
            return others[int(rng.integers(len(others)))] if others else truth[i]
        return truth[i]

    first = max(required_sample(size, THETA, ALPHA_STAGE, 0), 1)
    asked = [int(i) for i in order[:first]]
    answers = [answer(i) for i in asked]
    top, _, answered, need, passed, may_top_up = _look(answers, size)
    if may_top_up:
        more = [int(i) for i in order[len(asked) : len(asked) + (need - answered)]]
        if more:
            asked += more
            answers += [answer(i) for i in more]
            top, _, _, _, passed, _ = _look(answers, size)
    if not passed:
        return ClusterResult(size, size, False, top, ())
    seen = set(asked)
    return ClusterResult(
        size, len(asked), True, top, tuple(i for i in range(size) if i not in seen)
    )


def _members_by_cluster(codes: NDArray[np.integer[Any]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i, c in enumerate(codes.tolist()):
        out.setdefault(int(c), []).append(i)
    return out


def simulate(
    codes: NDArray[np.integer[Any]],
    truth: Sequence[str],
    *,
    seed: int,
    open_set: Iterable[str] = (),
    noise: float = 0.0,
) -> dict[str, Any]:
    """The gate over every cluster of ``codes`` (one code per pool item) at one sampling seed.

    Each cluster draws from its own generator seeded by ``(seed, code)``, so the result does not
    depend on the order the clusters are visited.
    """
    t = np.asarray(truth).astype(str)
    classes = sorted(set(t.tolist()))
    opened = set(open_set)
    cost = passed_clusters = propagated = correct = open_prop = open_correct = 0
    for code, rows in sorted(_members_by_cluster(codes).items()):
        rng = np.random.default_rng([seed, code & 0xFFFFFFFF])
        res = run_gate([t[i] for i in rows], rng, noise=noise, classes=classes)
        cost += res.sampled
        passed_clusters += res.passed
        for pos in res.propagated:
            hit = t[rows[pos]] == res.top
            propagated += 1
            correct += hit
            if t[rows[pos]] in opened:
                open_prop += 1
                open_correct += hit
    n = len(t)
    return {
        'items': n,
        'clusters': len(set(codes.tolist())),
        'passed_clusters': passed_clusters,
        'decisions': cost,
        'u1': cost / n if n else math.nan,
        'propagated': propagated,
        'u2': correct / propagated if propagated else math.nan,
        'u3': propagated / n if n else math.nan,
        'open_propagated': open_prop,
        'open_correct': open_correct,
        'u2_open': open_correct / open_prop if open_prop else math.nan,
    }


def seeded_runs(
    codes: NDArray[np.integer[Any]],
    truth: Sequence[str],
    *,
    seeds: Sequence[int],
    open_set: Iterable[str] = (),
    noise: float = 0.0,
) -> dict[str, Any]:
    """``simulate`` at every seed; each metric as mean, min and max over the seeds."""
    runs = [simulate(codes, truth, seed=s, open_set=open_set, noise=noise) for s in seeds]
    out: dict[str, Any] = {'seeds': list(seeds), 'noise': noise}
    for key in ('u1', 'u2', 'u3', 'u2_open', 'propagated', 'open_propagated', 'passed_clusters'):
        vals = [r[key] for r in runs if not (isinstance(r[key], float) and math.isnan(r[key]))]
        out[key] = (
            {'mean': float(np.mean(vals)), 'min': float(min(vals)), 'max': float(max(vals))}
            if vals
            else None
        )
    out['items'] = runs[0]['items']
    out['clusters'] = runs[0]['clusters']
    return out


def gate_reach(sizes: Iterable[int]) -> dict[str, Any]:
    """How many of the clusters could ever pass the gate: those larger than the sample they need.

    A cluster of ``n`` members is asked whole when ``required_sample(n, ...) == n``, so only
    ``n`` above the no-dissent sample can propagate anything. The "at most half" count is the
    clusters whose sample is half of them or less, where propagation pays for itself clearly.
    """
    sz = sorted(sizes)
    able = [n for n in sz if required_sample(n, THETA, ALPHA_STAGE, 0) < n]
    cheap = [n for n in sz if 2 * required_sample(n, THETA, ALPHA_STAGE, 0) <= n]
    return {
        'clusters': len(sz),
        'largest': sz[-1] if sz else 0,
        'clusters_that_can_propagate': len(able),
        'items_in_them': sum(able),
        'items': sum(sz),
        'smallest_cluster_that_can_propagate': min(able) if able else None,
        'clusters_where_the_sample_is_at_most_half': len(cheap),
        'items_in_them_at_most_half': sum(cheap),
    }
