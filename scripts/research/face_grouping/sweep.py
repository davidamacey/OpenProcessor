"""Run every grouping method over a threshold sweep on one test bed and score each point."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.research.face_grouping import grouping as gp, scores as sc


SIMS = [round(s, 2) for s in np.arange(0.30, 0.801, 0.05)]
WIDE_SIMS = [round(s, 2) for s in np.arange(0.10, 0.801, 0.05)]
GRAPH_K = 30
GRAPH_FLOOR = 0.30


def run_spec(spec: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """One method at one parameter point, from scratch on ``x`` (used by the gate and the incremental run)."""
    m, p = spec['method'], spec['params']
    if len(x) < 2:
        return np.zeros(len(x), dtype=np.int64)
    if m in ('average', 'complete'):
        return gp.agglomerative(x, m, p['sim'])
    g = gp.build_graph(x, k=GRAPH_K, floor=GRAPH_FLOOR)
    if m == 'single':
        return gp.single_linkage(g, p['sim'])
    if m == 'immich':
        return gp.immich_core(g, p['max_distance'], p['min_faces'])
    if m == 'mutual_knn':
        return gp.mutual_knn(g, p['k'], p['floor'], p.get('merge_sim'))
    if m == 'chinese_whispers':
        return gp.chinese_whisper_groups(g, p['floor'], seed=0)
    raise ValueError(m)


def join_sim(spec: dict[str, Any]) -> float:
    """Cosine at which a later face may join an existing cluster for a method's parameter point."""
    p = spec['params']
    if 'sim' in p:
        return float(p['sim'])
    if 'max_distance' in p:
        return 1.0 - float(p['max_distance'])
    return float(p.get('floor', 0.5))


def sweep(x: np.ndarray, truth: np.ndarray, min_images: int = 10) -> list[dict[str, Any]]:
    g = gp.build_graph(x, k=GRAPH_K, floor=GRAPH_FLOOR)
    rows: list[dict[str, Any]] = []

    def add(method: str, params: dict[str, Any], labels: np.ndarray) -> None:
        rows.append(
            {'method': method, 'params': params, **sc.identity_report(truth, labels, min_images)}
        )

    for s in SIMS:
        add('single', {'sim': s}, gp.single_linkage(g, s))
    for method in ('average', 'complete'):
        tree = gp.linkage_tree(x, method)
        for s in WIDE_SIMS:
            add(method, {'sim': s}, gp.cut_tree(tree, s))
    for s in SIMS:
        for mf in (2, 3, 5):
            add(
                'immich',
                {'max_distance': round(1 - s, 2), 'min_faces': mf},
                gp.immich_core(g, 1 - s, mf),
            )
    for k in (10, 20, 30):
        for floor in (0.3, 0.4, 0.5, 0.6):
            add('mutual_knn', {'k': k, 'floor': floor}, gp.mutual_knn(g, k, floor, None))
        for floor in (0.4, 0.5, 0.6):
            for merge in (0.35, 0.5):
                add(
                    'mutual_knn',
                    {'k': k, 'floor': floor, 'merge_sim': merge},
                    gp.mutual_knn(g, k, floor, merge),
                )
    for s in SIMS:
        add('chinese_whispers', {'floor': s}, gp.chinese_whisper_groups(g, s, seed=0))
    for level, labels in enumerate(gp.finch_levels(x)):
        add('finch', {'level': level}, labels)
    return rows
