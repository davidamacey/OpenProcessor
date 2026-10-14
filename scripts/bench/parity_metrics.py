"""Pure parity metrics: box matching, vector agreement, clustering agreement, strings.

Every function takes plain numpy arrays or lists and returns plain numbers, so a
synthetic fixture pins each one down. Nothing here touches the network or disk.
The gate thresholds live in ``parity_gates.json``; this module only measures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

MATCH_IOU = 0.5
IDENTICAL_IOU = 0.95
SCORE_TOLERANCE = 0.02
CONFIDENT_SCORE = 0.5


# ---------------------------------------------------------------- boxes


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix of two ``(n, 4)`` / ``(m, 4)`` arrays of ``x1, y1, x2, y2`` boxes."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    lt = np.maximum(a64[:, None, :2], b64[None, :, :2])
    rb = np.minimum(a64[:, None, 2:], b64[None, :, 2:])
    inter = np.clip(rb - lt, 0.0, None).prod(axis=2)
    area_a = np.clip(a64[:, 2:] - a64[:, :2], 0.0, None).prod(axis=1)
    area_b = np.clip(b64[:, 2:] - b64[:, :2], 0.0, None).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_boxes(
    ref_boxes: np.ndarray,
    ref_cls: Sequence[str] | np.ndarray,
    cand_boxes: np.ndarray,
    cand_cls: Sequence[str] | np.ndarray,
    min_iou: float = MATCH_IOU,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy one-to-one matching, best IoU first, only within the same class name.

    Returns ``(pairs, unmatched_ref, unmatched_cand)`` as indices into the inputs.
    """
    if len(ref_boxes) == 0 or len(cand_boxes) == 0:
        return [], list(range(len(ref_boxes))), list(range(len(cand_boxes)))
    iou = box_iou(ref_boxes, cand_boxes)
    same = np.asarray(ref_cls)[:, None] == np.asarray(cand_cls)[None, :]
    iou = np.where(same, iou, 0.0)
    ri, ci = np.nonzero(iou >= min_iou)
    order = np.lexsort((ci, ri, -iou[ri, ci]))
    used_r: set[int] = set()
    used_c: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for k in order:
        r, c = int(ri[k]), int(ci[k])
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        pairs.append((r, c))
    miss_ref = [i for i in range(len(ref_boxes)) if i not in used_r]
    miss_cand = [i for i in range(len(cand_boxes)) if i not in used_c]
    return sorted(pairs), miss_ref, miss_cand


def _group(images: np.ndarray) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(images.tolist()):
        groups[name].append(i)
    return groups


def _match_by_image(
    ref: tuple[Any, ...], cand: tuple[Any, ...], min_iou: float
) -> list[tuple[str, list[tuple[int, int]], list[int], list[int]]]:
    """Per image (sorted by name): matched pairs and the unmatched ref / cand indices.

    ``ref`` and ``cand`` are ``(image, box, cls, ...)``; indices are global.
    """
    ref_groups, cand_groups = _group(ref[0]), _group(cand[0])
    out = []
    for name in sorted(set(ref_groups) | set(cand_groups)):
        ri, ci = ref_groups.get(name, []), cand_groups.get(name, [])
        pairs, miss_r, miss_c = match_boxes(
            ref[1][ri].reshape(-1, 4), ref[2][ri], cand[1][ci].reshape(-1, 4), cand[2][ci], min_iou
        )
        out.append(
            (
                name,
                [(ri[a], ci[b]) for a, b in pairs],
                [ri[a] for a in miss_r],
                [ci[b] for b in miss_c],
            )
        )
    return out


def pair_items(
    ref: tuple[np.ndarray, np.ndarray, np.ndarray],
    cand: tuple[np.ndarray, np.ndarray, np.ndarray],
    min_iou: float = MATCH_IOU,
) -> np.ndarray:
    """``(P, 2)`` global index pairs ``(ref_i, cand_i)`` of items matched by image, box and class."""
    rows = [p for _, pairs, _, _ in _match_by_image(ref, cand, min_iou) for p in pairs]
    return np.array(sorted(rows), dtype=np.int64).reshape(-1, 2)


def box_metrics(
    ref: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cand: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    min_iou: float = MATCH_IOU,
    identical_iou: float = IDENTICAL_IOU,
    score_tolerance: float = SCORE_TOLERANCE,
    confident: float = CONFIDENT_SCORE,
) -> dict[str, Any]:
    """P-box and P-item numbers from ``(image, box, class, score)`` tuples of two captures."""
    ref_score, cand_score = ref[3], cand[3]
    per_image = _match_by_image(ref, cand, min_iou)
    ref_cls_by_image = {k: sorted(ref[2][v].tolist()) for k, v in _group(ref[0]).items()}
    cand_cls_by_image = {k: sorted(cand[2][v].tolist()) for k, v in _group(cand[0]).items()}
    identical = same_items = unmatched_confident = matched = 0
    ious: list[float] = []
    deltas: list[float] = []
    for name, pairs, miss_r, miss_c in per_image:
        pair_iou = [float(box_iou(ref[1][r : r + 1], cand[1][c : c + 1])[0, 0]) for r, c in pairs]
        pair_delta = [abs(float(ref_score[r]) - float(cand_score[c])) for r, c in pairs]
        ious.extend(pair_iou)
        deltas.extend(pair_delta)
        matched += len(pairs)
        unmatched_confident += sum(1 for r in miss_r if ref_score[r] >= confident)
        unmatched_confident += sum(1 for c in miss_c if cand_score[c] >= confident)
        if ref_cls_by_image.get(name, []) == cand_cls_by_image.get(name, []):
            same_items += 1
        if (
            not miss_r
            and not miss_c
            and all(v >= identical_iou for v in pair_iou)
            and all(d <= score_tolerance for d in pair_delta)
        ):
            identical += 1
    n_images = len(per_image)
    return {
        'images': n_images,
        'items_ref': len(ref[0]),
        'items_cand': len(cand[0]),
        'matched': matched,
        'images_identical_share': identical / n_images if n_images else 1.0,
        'item_count_class_identical_share': same_items / n_images if n_images else 1.0,
        'unmatched_confident': unmatched_confident,
        'iou_min': min(ious) if ious else None,
        'iou_mean': float(np.mean(ious)) if ious else None,
        'score_delta_max': max(deltas) if deltas else None,
    }


# ---------------------------------------------------------------- vectors


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of two ``(n, d)`` arrays (computed in float64)."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    denom = np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1)
    return np.divide((a64 * b64).sum(axis=1), denom, out=np.zeros(len(a64)), where=denom > 0)


def cosine_summary(cos: np.ndarray) -> dict[str, Any]:
    if len(cos) == 0:
        return {'n': 0}
    return {
        'n': len(cos),
        'min': float(cos.min()),
        'mean': float(cos.mean()),
        'p01': float(np.percentile(cos, 1)),
        'median': float(np.median(cos)),
    }


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    norm = np.linalg.norm(x32, axis=1, keepdims=True)
    return x32 / np.where(norm > 0, norm, 1.0)


def knn_indices(x: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` cosine neighbours (self excluded) of each query row, best first."""
    xn = _unit_rows(x)
    sims = xn[queries] @ xn.T
    sims[np.arange(len(queries)), queries] = -np.inf
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1, kind='stable')
    return np.take_along_axis(top, order, axis=1)


def knn_agreement(
    ref: np.ndarray,
    cand: np.ndarray,
    *,
    k: int,
    n_queries: int,
    seed: int,
    dup_threshold: float,
    labels: np.ndarray | None = None,
    truth: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Neighbour-set agreement of two row-aligned vector sets.

    ``overlap_at_k`` is the mean share of the reference top-k found in the candidate top-k;
    ``top1_recall_at_k`` the share of queries whose reference nearest neighbour is in the
    candidate top-k; ``near_dup_agreement`` the share of query/neighbour pairs (from either
    list) that sit on the same side of ``dup_threshold`` in both spaces.

    With ``labels`` (one per row) it adds ``label_vote_agreement``: the share of queries whose
    majority label among the top-k is the same in both spaces (what label propagation would
    do). With ``truth`` (reference and candidate ground-truth class per row) it adds the kNN
    class precision of each space and ``precision_drop_points``; queries without a ground-truth
    class are left out.
    """
    n = len(ref)
    k = min(k, n - 1)
    if k < 1:
        return {'queries': 0, 'k': 0}
    queries = np.sort(np.random.default_rng(seed).choice(n, size=min(n_queries, n), replace=False))
    ref_nn, cand_nn = knn_indices(ref, queries, k), knn_indices(cand, queries, k)
    overlap = [
        len(set(r) & set(c)) / k for r, c in zip(ref_nn.tolist(), cand_nn.tolist(), strict=True)
    ]
    top1 = [int(r[0] in set(c)) for r, c in zip(ref_nn.tolist(), cand_nn.tolist(), strict=True)]
    ref_n, cand_n = _unit_rows(ref), _unit_rows(cand)
    qs: list[int] = []
    ns: list[int] = []
    for q, r, c in zip(queries.tolist(), ref_nn.tolist(), cand_nn.tolist(), strict=True):
        for m in sorted(set(r) | set(c)):
            qs.append(q)
            ns.append(m)
    q_idx, n_idx = np.array(qs), np.array(ns)
    in_ref = (ref_n[q_idx] * ref_n[n_idx]).sum(axis=1) >= dup_threshold
    in_cand = (cand_n[q_idx] * cand_n[n_idx]).sum(axis=1) >= dup_threshold
    out: dict[str, Any] = {
        'queries': len(queries),
        'k': k,
        'overlap_at_k': float(np.mean(overlap)),
        'top1_recall_at_k': float(np.mean(top1)),
        'near_dup_pairs': len(q_idx),
        'near_dup_agreement': float(np.mean(in_ref == in_cand)),
    }
    if labels is not None:
        votes = [(_vote(labels[r]), _vote(labels[c])) for r, c in zip(ref_nn, cand_nn, strict=True)]
        out['label_vote_agreement'] = float(np.mean([a == b for a, b in votes]))
    if truth is not None:
        p_ref = _neighbour_precision(truth[0], queries, ref_nn)
        p_cand = _neighbour_precision(truth[1], queries, cand_nn)
        if p_ref is not None and p_cand is not None:
            out['precision_ref'] = p_ref
            out['precision_cand'] = p_cand
            out['precision_drop_points'] = (p_ref - p_cand) * 100.0
    return out


def _vote(row: np.ndarray) -> str:
    """The most common label of a neighbour list; ties go to the alphabetically first."""
    counts = Counter(row.tolist())
    return min(counts, key=lambda label: (-counts[label], label))


def _neighbour_precision(truth: np.ndarray, queries: np.ndarray, nn: np.ndarray) -> float | None:
    """Mean share of a query's neighbours that share its ground-truth class (classed queries only)."""
    known = [i for i, q in enumerate(queries.tolist()) if truth[q] not in ('', 'none')]
    if not known:
        return None
    return float(np.mean([np.mean(truth[nn[i]] == truth[queries[i]]) for i in known]))


# ---------------------------------------------------------------- clusters


def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    return table


def _comb2(x: np.ndarray | int) -> Any:
    return x * (x - 1) / 2


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index of two labellings (invariant to label renaming)."""
    table = _contingency(a, b)
    n = int(table.sum())
    sum_cells = float(_comb2(table).sum())
    sum_a = float(_comb2(table.sum(axis=1)).sum())
    sum_b = float(_comb2(table.sum(axis=0)).sum())
    expected = sum_a * sum_b / _comb2(n) if n > 1 else 0.0
    maximum = (sum_a + sum_b) / 2
    return 1.0 if maximum == expected else (sum_cells - expected) / (maximum - expected)


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def normalized_mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    """NMI with the arithmetic-mean normalisation."""
    table = _contingency(a, b).astype(np.float64)
    n = table.sum()
    pa, pb = table.sum(axis=1), table.sum(axis=0)
    nz = table > 0
    outer = np.outer(pa, pb)
    mi = float((table[nz] / n * np.log(table[nz] * n / outer[nz])).sum())
    denom = (_entropy(pa) + _entropy(pb)) / 2
    return 1.0 if denom == 0 else mi / denom


def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Size-weighted share of each cluster's modal ground-truth class."""
    if len(labels) == 0:
        return 0.0
    table = _contingency(labels, truth)
    return float(table.max(axis=1).sum() / table.sum())


def _centroids(vec: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unit = _unit_rows(vec)
    ids = np.unique(labels)
    cents = np.stack([unit[labels == i].mean(axis=0) for i in ids])
    return _unit_rows(cents)


def assignment_agreement(
    ref_vec: np.ndarray, cand_vec: np.ndarray, ref_labels: np.ndarray
) -> float:
    """Share of items whose nearest reference centroid is the same for ref and candidate vectors.

    The reference centroids are the normalised mean of each reference cluster's vectors
    (the API does not serve centroids), so this isolates the effect of the vector change
    from k-means' own run-to-run variation.
    """
    cents = _centroids(ref_vec, ref_labels)
    ref_near = np.argmax(_unit_rows(ref_vec) @ cents.T, axis=1)
    cand_near = np.argmax(_unit_rows(cand_vec) @ cents.T, axis=1)
    return float(np.mean(ref_near == cand_near))


def kmeans_objective(vec: np.ndarray, labels: np.ndarray) -> float:
    """Mean cosine distance of each vector to the normalised mean of its own cluster."""
    unit = _unit_rows(vec)
    _, inverse = np.unique(labels, return_inverse=True)
    cents = _centroids(vec, labels)
    return float(np.mean(1.0 - (unit * cents[inverse]).sum(axis=1)))


# ---------------------------------------------------------------- strings


def agreement_share(a: Sequence[str], b: Sequence[str]) -> float:
    """Share of positions where two equal-length string lists agree (1.0 when empty)."""
    if not a:
        return 1.0
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_agreement(
    ref: tuple[np.ndarray, np.ndarray, np.ndarray],
    cand: tuple[np.ndarray, np.ndarray, np.ndarray],
    min_iou: float = MATCH_IOU,
) -> dict[str, Any]:
    """Exact-match share and character error rate of OCR lines matched by box overlap.

    A line present in only one capture counts against ``exact_share``.

    ``ref`` and ``cand`` are ``(image, box, text)``. Lines are matched per image at
    ``min_iou`` regardless of their text.
    """
    same_kind = np.array(['text'] * len(ref[0])), np.array(['text'] * len(cand[0]))
    matches = _match_by_image(
        (ref[0], ref[1], same_kind[0]), (cand[0], cand[1], same_kind[1]), min_iou
    )
    pairs = [p for _, ps, _, _ in matches for p in ps]
    exact = sum(ref[2][r] == cand[2][c] for r, c in pairs)
    chars = sum(len(str(ref[2][r])) for r, _ in pairs)
    errors = sum(_edit_distance(str(ref[2][r]), str(cand[2][c])) for r, c in pairs)
    lines = max(len(ref[0]), len(cand[0]))
    return {
        'lines_ref': len(ref[0]),
        'lines_cand': len(cand[0]),
        'matched': len(pairs),
        'exact_share': exact / lines if lines else 1.0,
        'char_error_rate': errors / chars if chars else 0.0,
    }
