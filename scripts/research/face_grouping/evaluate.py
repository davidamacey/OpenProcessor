"""Parts A, D and E of the face-grouping evaluation (verification, quality signals, incremental stability)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench.cluster_eval_metrics import moved_items
from scripts.bench.parity_metrics import adjusted_rand_index
from scripts.research.face_grouping import (
    datasets as ds,
    face_table as ft,
    incremental as inc,
    quality as q,
    scores as sc,
    sweep as sw,
    verification as ver,
)


if TYPE_CHECKING:
    from pathlib import Path


def set_pairs(eval_dir: Path, name: str) -> list[tuple[str, str, int]]:
    text = (eval_dir / name / ds.PAIRS[name]).read_text()
    return ft.parse_lfw_pairs(text) if name == 'lfw' else ft.parse_cross_pairs(text)


def pair_table(s: ds.FaceSet, pairs: list[tuple[str, str, int]]) -> dict[str, Any]:
    """Cosine score per pair; a pair with a missing face scores -1 (never accepted) and is counted."""
    index = {b: i for i, b in enumerate(s.basename)}
    ia = np.array([index[a] for a, _, _ in pairs])
    ib = np.array([index[b] for _, b, _ in pairs])
    ok = s.has[ia] & s.has[ib]
    score = np.where(ok, (s.emb[ia] * s.emb[ib]).sum(axis=1), -1.0)
    return {'ia': ia, 'ib': ib, 'ok': ok, 'score': score, 'label': np.array([p[2] for p in pairs])}


def verification_report(t: dict[str, Any]) -> dict[str, Any]:
    folds = ft.fold_ids(len(t['label']))
    acc = ver.fold_accuracy(t['score'], t['label'], folds)
    return {
        'pairs': len(t['label']),
        'pairs_missing_a_face': int((~t['ok']).sum()),
        **acc,
        'tar_at_far_1e-2': ver.tar_at_far(t['score'], t['label'], folds, 1e-2),
        'tar_at_far_1e-3': ver.tar_at_far(t['score'], t['label'], folds, 1e-3),
    }


def pair_errors(t: dict[str, Any]) -> np.ndarray:
    """1 where the pair is wrong at the accuracy-maximising threshold over all of the set's pairs."""
    thr = ver.best_threshold(t['score'], t['label'])
    wrong: np.ndarray = (t['score'] >= thr) != (t['label'] == 1)
    return wrong.astype(np.int64)


def quality_report(
    pooled: ds.Pooled, tables: dict[str, dict[str, Any]], partition: np.ndarray | None
) -> dict[str, Any]:
    """Error-versus-reject area per signal for verification pairs and for wrong-merged faces."""
    out: dict[str, Any] = {}
    err_rows: list[np.ndarray] = []
    signal_rows: dict[str, list[np.ndarray]] = {k: [] for k in pooled.signals}
    for name, t in tables.items():
        keep = t['ok']
        ra, rb = ds_rows(pooled, name, t)
        err_rows.append(pair_errors(t)[keep])
        for k, v in pooled.signals.items():
            signal_rows[k].append(np.minimum(v[ra[keep]], v[rb[keep]]))
    err = np.concatenate(err_rows)
    wrong = None if partition is None else minority_faces(pooled.identity, partition)
    for k, parts in signal_rows.items():
        entry: dict[str, Any] = {
            'pair_error_reject_area': q.error_reject_area(np.concatenate(parts), err),
            'pair_base_error': float(err.mean()),
        }
        if wrong is not None:
            entry['wrong_merge_reject_area'] = q.error_reject_area(pooled.signals[k], wrong)
            entry['wrong_merge_base_rate'] = float(wrong.mean())
        out[k] = entry
    return out


def ds_rows(pooled: ds.Pooled, name: str, t: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return pooled.row_of[name][t['ia']], pooled.row_of[name][t['ib']]


def minority_faces(identity: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    """1 for a face that sits in a multi-face cluster whose majority identity is another one."""
    t = np.unique(identity, return_inverse=True)[1]
    c = sc.with_singletons(cluster)
    cell, n = np.unique(c.astype(np.int64) * (t.max() + 1) + t, return_counts=True)
    cc, ct = cell // (t.max() + 1), cell % (t.max() + 1)
    major = np.zeros(c.max() + 1, dtype=np.int64)
    top = np.zeros(c.max() + 1)
    for ci, ti, ni in zip(cc, ct, n, strict=True):
        if ni > top[ci]:
            top[ci], major[ci] = ni, ti
    size = np.bincount(c)
    minority: np.ndarray = (t != major[c]) & (size[c] >= 2)
    return minority.astype(np.int64)


def gated_report(
    pooled: ds.Pooled, spec: dict[str, Any], signal: str, cuts: tuple[float, ...] = (0.1, 0.2, 0.3)
) -> list[dict[str, Any]]:
    sig = pooled.signals[signal]
    rows = []
    for frac in (0.0, *cuts):
        cut = float(np.quantile(sig, frac)) if frac else -np.inf
        labels = q.gated(
            pooled.emb, sig, cut, lambda sub: sw.run_spec(spec, sub), sw.join_sim(spec)
        )
        rows.append(
            {
                'reject_fraction': frac,
                'signal': signal,
                **sc.identity_report(pooled.identity, labels),
            }
        )
    return rows


def incremental_report(
    pooled: ds.Pooled, spec: dict[str, Any], batches: int = 10
) -> dict[str, Any]:
    n = len(pooled.emb)
    order = np.random.default_rng(0).permutation(n)
    parts = np.array_split(order, batches)

    def method(sub: np.ndarray) -> np.ndarray:
        return sw.run_spec(spec, sub)

    steps = list(inc.incremental_steps(pooled.emb, parts, method, sw.join_sim(spec)))
    one_shot = method(pooled.emb)
    final = steps[-1]
    before = np.concatenate(parts[:-1])
    protected = np.zeros(n, dtype=bool)
    protected[before] = True

    def scratch(rows: np.ndarray) -> np.ndarray:
        out = np.full(n, -1, dtype=np.int64)
        out[rows] = method(pooled.emb[rows])
        return out

    scratch9, scratch10 = scratch(before), scratch(order)
    first = sc.with_singletons(final)
    return {
        'spec': spec,
        'batches': batches,
        'incremental_scores': sc.identity_report(pooled.identity, final),
        'one_shot_scores': sc.identity_report(pooled.identity, one_shot),
        'ari_incremental_vs_one_shot': adjusted_rand_index(first, sc.with_singletons(one_shot)),
        'moved_share_incremental_batch9_to_10': moved_items(steps[-2], final, protected)
        / len(before),
        'moved_share_recluster_batch9_to_10': moved_items(scratch9, scratch10, protected)
        / len(before),
    }
