"""Pure logic of the vector store benchmark: filters, recall, percentiles, ground truth.

No engine client is imported here. Everything that decides what a result means (which rows a
filter keeps, what recall is, how a percentile is read, what the exact answer is) lives in this
module so ``tests/test_vdb_bench_lib.py`` can pin it without any service running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


# Selectivity sweep: ``sel`` is a uniform integer in [0, SEL_RANGE); a filter ``sel < n`` keeps n/SEL_RANGE.
SEL_RANGE = 10_000
PROJECT_COUNT = 10
FIELD_SHARES = {'class_validated': 0.10, 'test_holdout': 0.05, 'review_dismissed': 0.02}
CLUSTER_COUNT = 512


@dataclass(frozen=True)
class FilterSpec:
    """One filter of the matrix, expressed as plain data every adapter translates itself.

    ``eq`` fields must equal the value, ``is_false`` booleans must be false, ``sel_lt`` keeps
    rows with ``sel`` below the bound.
    """

    name: str
    eq: tuple[tuple[str, Any], ...] = ()
    is_false: tuple[str, ...] = ()
    sel_lt: int | None = None


def filter_matrix(project: int, class_name: str, cluster_id: int) -> list[FilterSpec]:
    """The filters of the plan: none, project, project+class, review default, cluster, sweep."""
    proj = ('project', project)
    return [
        FilterSpec('none'),
        FilterSpec('project', eq=(proj,)),
        FilterSpec('project_class', eq=(proj, ('class_name', class_name))),
        FilterSpec(
            'review_default',
            eq=(proj,),
            is_false=('class_validated', 'test_holdout', 'review_dismissed'),
        ),
        FilterSpec('project_cluster', eq=(proj, ('cluster_id', cluster_id))),
        FilterSpec('sel50', sel_lt=SEL_RANGE // 2),
        FilterSpec('sel10', sel_lt=SEL_RANGE // 10),
        FilterSpec('sel1', sel_lt=SEL_RANGE // 100),
        FilterSpec('sel01', sel_lt=SEL_RANGE // 1000),
    ]


def filter_mask(spec: FilterSpec, meta: dict[str, np.ndarray]) -> np.ndarray:
    """Boolean mask of the rows ``spec`` keeps."""
    n = len(meta['sel'])
    mask = np.ones(n, dtype=bool)
    for field, value in spec.eq:
        mask &= meta[field] == value
    for field in spec.is_false:
        mask &= ~meta[field].astype(bool)
    if spec.sel_lt is not None:
        mask &= meta['sel'] < spec.sel_lt
    return mask


def make_filter_fields(
    image_ids: np.ndarray, item_ids: np.ndarray, seed: int
) -> dict[str, np.ndarray]:
    """Seeded synthetic filter fields: project by image hash, flags and ``sel`` per item."""
    project = (image_ids.astype(np.int64) * 2654435761 % (2**32)) % PROJECT_COUNT
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {'project': project.astype(np.int64)}
    for name, share in FIELD_SHARES.items():
        out[name] = rng.random(len(item_ids)) < share
    out['sel'] = rng.integers(0, SEL_RANGE, len(item_ids)).astype(np.int64)
    return out


def exact_topk(
    vectors: np.ndarray, queries: np.ndarray, mask: np.ndarray, k: int, block: int = 256
) -> np.ndarray:
    """Exact cosine top-``k`` row indexes per query among the rows ``mask`` keeps.

    Vectors and queries must be L2-normalized. Rows are returned best first; a query with
    fewer than ``k`` matches is padded with -1.
    """
    rows = np.flatnonzero(mask)
    out = np.full((len(queries), k), -1, dtype=np.int64)
    if len(rows) == 0:
        return out
    sub = np.ascontiguousarray(vectors[rows])
    kk = min(k, len(rows))
    for start in range(0, len(queries), block):
        scores = queries[start : start + block] @ sub.T
        part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        order = np.take_along_axis(scores, part, axis=1).argsort(axis=1)[:, ::-1]
        out[start : start + block, :kk] = rows[np.take_along_axis(part, order, axis=1)]
    return out


def recall_at_k(truth: np.ndarray, found: list[list[int]], k: int) -> float:
    """Mean over queries of |found[:k] and truth[:k]| / |truth[:k] without padding|.

    Queries whose exact answer is empty are skipped (nothing to recall).
    """
    scores = []
    for t_row, f_row in zip(truth, found, strict=True):
        want = {int(x) for x in t_row[:k] if x >= 0}
        if not want:
            continue
        scores.append(len(want & {int(x) for x in f_row[:k]}) / len(want))
    return float(np.mean(scores)) if scores else float('nan')


def short_rate(found: list[list[int]], truth: np.ndarray, k: int) -> float:
    """Share of queries that returned fewer rows than the exact answer has (up to ``k``)."""
    short = 0
    for t_row, f_row in zip(truth, found, strict=True):
        want = int((t_row[:k] >= 0).sum())
        if len(f_row) < want:
            short += 1
    return short / len(found) if len(found) else float('nan')


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (the smallest value covering ``pct`` percent)."""
    if not values:
        return float('nan')
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def latency_summary(ms: list[float]) -> dict[str, float]:
    return {
        'p50_ms': percentile(ms, 50),
        'p95_ms': percentile(ms, 95),
        'p99_ms': percentile(ms, 99),
        'mean_ms': float(np.mean(ms)) if ms else float('nan'),
    }


def median_range(values: list[float]) -> tuple[float, float, float]:
    """(median, min, max) of the repetitions."""
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)), float(arr.min()), float(arr.max())


def bytes_per_vector(total_bytes: int, vectors: int) -> float:
    return total_bytes / vectors if vectors else float('nan')
