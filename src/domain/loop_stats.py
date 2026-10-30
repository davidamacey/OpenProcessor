"""Exact one-sided bounds behind the auto-label loop's propagation gate: pure,
stdlib only.

A node has ``N`` eligible members. A share ``h`` of them would get class ``c``
from the VLM. Sampling ``n`` without replacement and seeing ``x`` votes for
``c`` is evidence for ``h >= theta`` only when, were ``h`` just below
``theta`` (``K = ceil(theta * N) - 1`` agreeing members), seeing at least ``x``
would be rare: ``P(X >= x | N, K, n) <= alpha`` (hypergeometric). For a large
``N`` this is the one-sided Clopper-Pearson bound. It bounds agreement with the
VLM, not correctness."""

from __future__ import annotations

import math


def _tail(n_members: int, agreeing: int, n: int, at_least: int) -> float:
    """``P(X >= at_least)`` for ``X`` agreeing members in a sample of ``n`` drawn
    without replacement from ``n_members`` of which ``agreeing`` agree."""
    total = math.comb(n_members, n)
    hit = sum(
        math.comb(agreeing, k) * math.comb(n_members - agreeing, n - k)
        for k in range(max(at_least, 0), min(n, agreeing) + 1)
    )
    return hit / total


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        msg = 'alpha must be between 0 and 1 (exclusive)'
        raise ValueError(msg)


def required_sample(n_members: int, theta: float, alpha: float, dissent: int) -> int:
    """Smallest sample size ``n`` for which ``n - dissent`` agreeing answers prove
    ``h >= theta`` at level ``alpha``. A node too small to ever prove it is asked
    whole: ``n_members`` is returned, which is then an exact answer."""
    if n_members < 1:
        msg = 'n_members must be at least 1'
        raise ValueError(msg)
    if not 0.0 < theta < 1.0:
        msg = 'theta must be between 0 and 1 (exclusive)'
        raise ValueError(msg)
    _check_alpha(alpha)
    if dissent < 0:
        msg = 'dissent must not be negative'
        raise ValueError(msg)
    below = math.ceil(theta * n_members) - 1
    for n in range(dissent + 1, n_members):
        if _tail(n_members, below, n, n - dissent) <= alpha:
            return n
    return n_members


def homogeneity_lower_bound(agree: int, n: int, n_members: int, alpha: float) -> float:
    """Exact one-sided ``1 - alpha`` lower bound on the share of ``n_members``
    that agree, given ``agree`` of ``n`` sampled did. It is ``K / n_members`` for
    the smallest ``K`` that ``agree`` agreements does not reject, so the gate
    passes at ``theta`` exactly when this bound is at least ``theta``."""
    if not 0 <= agree <= n <= n_members:
        msg = 'counts must satisfy 0 <= agree <= n <= n_members'
        raise ValueError(msg)
    _check_alpha(alpha)
    low, high = agree, n_members - (n - agree)
    while low < high:
        mid = (low + high) // 2
        if _tail(n_members, mid, n, agree) > alpha:
            high = mid
        else:
            low = mid + 1
    return low / n_members


def wilson_lower_bound(x: int, n: int, z: float) -> float:
    """Wilson score lower bound of ``x`` successes in ``n`` trials at normal
    quantile ``z`` (1.645 is one-sided 95 %). Zero when ``n`` is zero."""
    if not 0 <= x <= n:
        msg = 'counts must satisfy 0 <= x <= n'
        raise ValueError(msg)
    if n == 0:
        return 0.0
    phat = x / n
    z2 = z * z
    centre = phat + z2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return max(0.0, (centre - margin) / (1 + z2 / n))
