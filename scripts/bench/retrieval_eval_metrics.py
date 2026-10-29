"""Ranking metrics and summaries for the retrieval baseline (no I/O)."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any


def rank_of(ranked_ids: list[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant id, None when absent."""
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant:
            return i
    return None


def single_relevant_metrics(ranks: list[int | None]) -> dict[str, float]:
    """recall@1/5/10, MRR@10 and nDCG@10 for one relevant item per query (binary gain)."""
    n = len(ranks) or 1

    def hit(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / n

    mrr = sum(1.0 / r for r in ranks if r is not None and r <= 10) / n
    ndcg = sum(1.0 / math.log2(r + 1) for r in ranks if r is not None and r <= 10) / n
    return {
        'recall@1': hit(1),
        'recall@5': hit(5),
        'recall@10': hit(10),
        'mrr@10': mrr,
        'ndcg@10': ndcg,
    }


def bootstrap_ci(
    ranks: list[int | None], metric: str, resamples: int = 1000, seed: int = 7
) -> list[float]:
    rng = random.Random(seed)
    n = len(ranks)
    vals = []
    for _ in range(resamples):
        sample = [ranks[rng.randrange(n)] for _ in range(n)]
        vals.append(single_relevant_metrics(sample)[metric])
    vals.sort()
    return [vals[int(0.025 * resamples)], vals[int(0.975 * resamples) - 1]]


def with_ci(ranks: list[int | None]) -> dict[str, Any]:
    base = single_relevant_metrics(ranks)
    out: dict[str, Any] = {k: {'value': v, 'ci95': bootstrap_ci(ranks, k)} for k, v in base.items()}
    found = [r for r in ranks if r is not None]
    out['not_found_in_depth'] = len(ranks) - len(found)
    out['median_rank_found'] = statistics.median(found) if found else None
    return out


def pct(values: list[float], q: float) -> float:
    if not values:
        return float('nan')
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[idx]


def latency_summary(ms: list[float]) -> dict[str, float | int]:
    return {
        'n': len(ms),
        'p50': pct(ms, 0.5),
        'p95': pct(ms, 0.95),
        'p99': pct(ms, 0.99),
        'mean': statistics.fmean(ms) if ms else float('nan'),
        'min': min(ms) if ms else float('nan'),
        'max': max(ms) if ms else float('nan'),
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
