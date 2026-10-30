"""Novel-candidate clusters: far from the validated examples of their own detector class.

A crop's novelty is 1 minus its mean cosine to its ``k`` nearest validated examples of the
class it was filed under (the half-fold's labelled crops whose label equals that class, other
source images only). A cluster's novelty is the mean over its members. A cluster is flagged
when it is above the ``q``-quantile of the novelty that the class's own validated examples
have, with ``q`` the lowest that keeps the flag precision above its bar on the fitting half.

Ground truth for "novel" is an open-set label: one the detector has no class for. Everything
here is scored on the half the thresholds were not chosen on.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench import cluster_eval_metrics as met
from scripts.research.cluster_diag.dataset import image_name
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from scripts.research.cluster_diag.dataset import Bundle

    Floats = NDArray[np.floating[Any]]
    Ints = NDArray[np.integer[Any]]
    Mask = NDArray[np.bool_]

NEIGHBOURS = 10
MIN_EXAMPLES = 20  # validated examples a class needs before its threshold is trusted
MIN_NOVEL = 10  # open-set crops a class needs before its AUROC is judged
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
AUROC_BAR = 0.90
FLAG_PRECISION_BAR = 0.80
SUGGESTION_PRECISION_BAR = 0.90
MIN_FLAGGED = 5
MIN_SUGGESTIONS = 5


def auroc(positive: Floats, negative: Floats) -> float | None:
    """Probability that a random positive outscores a random negative (ties count half)."""
    if not len(positive) or not len(negative):
        return None
    diff = positive[:, None] - negative[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def crop_novelty(
    vectors: Floats,
    keys: Any,
    groups: Sequence[str],
    example_rows: Ints,
    example_keys: Any,
    k: int = NEIGHBOURS,
) -> Floats:
    """1 minus the mean cosine to the ``k`` nearest examples of the crop's own key.

    Examples from the crop's own source image are skipped; a crop with none left, or whose
    key has no example, scores NaN.
    """
    out = np.full(len(vectors), np.nan)
    k_arr, ex_keys = np.asarray(keys), np.asarray(example_keys)
    g = np.asarray(groups)
    for name in sorted(set(ex_keys.tolist())):
        ex = example_rows[ex_keys == name]
        rows = np.nonzero(k_arr == name)[0]
        if not len(rows):
            continue
        sims = vectors[rows] @ vectors[ex].T
        sims[g[rows][:, None] == g[ex][None, :]] = -np.inf
        top = -np.sort(-sims, axis=1)[:, :k]
        finite = np.isfinite(top)
        count = finite.sum(1)
        mean = np.where(count > 0, np.where(finite, top, 0.0).sum(1) / np.maximum(count, 1), np.nan)
        out[rows] = 1.0 - mean
    return out


def cluster_novelty(part: Ints, crop_scores: Floats) -> Floats:
    """Each crop's cluster score: the mean of the finite crop novelties in its cluster."""
    out = np.full(len(part), np.nan)
    for cid in np.unique(part):
        members = part == cid
        vals = crop_scores[members]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[members] = vals.mean()
    return out


def own_class_thresholds(
    keys: Any, scores: Floats, own_rows: Ints, q: float, min_examples: int = MIN_EXAMPLES
) -> dict[str, float]:
    """Per key, the ``q``-quantile of the scores of crops whose label equals the key.

    A key with fewer than ``min_examples`` own-class crops gets no threshold (it abstains).
    """
    k_arr = np.asarray(keys)
    out: dict[str, float] = {}
    for name in sorted(set(k_arr[own_rows].tolist())):
        vals = scores[own_rows[k_arr[own_rows] == name]]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= min_examples:
            out[name] = float(np.quantile(vals, q))
    return out


def flagged(keys: Any, scores: Floats, thresholds: Mapping[str, float]) -> Mask:
    """True where a crop's score is above its key's threshold (no threshold: never flagged)."""
    cut = np.array([thresholds.get(str(k), np.inf) for k in keys])
    with np.errstate(invalid='ignore'):
        return np.asarray(np.isfinite(scores) & (scores > cut))


def flag_quality(is_flagged: Mask, is_novel: Mask) -> dict[str, float | int | None]:
    """Precision and recall of the flags against the open-set labels."""
    n_flag = int(is_flagged.sum())
    hit = int((is_flagged & is_novel).sum())
    return {
        'flagged': n_flag,
        'novel': int(is_novel.sum()),
        'precision': hit / n_flag if n_flag else None,
        'recall': hit / int(is_novel.sum()) if is_novel.any() else None,
    }


def choose_quantile(
    keys: Any,
    scores: Floats,
    own_rows: Ints,
    judged: Ints,
    is_novel: Mask,
    bar: float = FLAG_PRECISION_BAR,
) -> tuple[float | None, list[dict[str, Any]]]:
    """The lowest quantile whose flag precision on ``judged`` crops reaches ``bar``.

    ``judged`` are indices into the labelled arrays (``is_novel`` is aligned to them).
    Returns the quantile (``None`` if none qualifies) and the table of all candidates.
    """
    table, chosen = [], None
    for q in QUANTILES:
        th = own_class_thresholds(keys, scores, own_rows, q)
        row = {'q': q, **flag_quality(flagged(keys, scores, th)[judged], is_novel)}
        table.append(row)
        if chosen is None and row['flagged'] >= MIN_FLAGGED and (row['precision'] or 0) >= bar:  # type: ignore[operator]
            chosen = q
    return chosen, table


def per_key_auroc(
    keys: Any,
    scores: Floats,
    scored: Ints,
    truth: Any,
    key_set: set[str],
    examples: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """Per key: own-class crops against open-set crops filed under it, on the scored half."""
    k_arr, t = np.asarray(keys), np.asarray(truth)
    out = {}
    for name in sorted(set(k_arr[scored].tolist())):
        rows = scored[k_arr[scored] == name]
        own = scores[rows[t[rows] == name]]
        novel = scores[rows[~np.isin(t[rows], list(key_set))]]
        own, novel = own[np.isfinite(own)], novel[np.isfinite(novel)]
        if not len(novel):
            continue
        out[name] = {
            'examples': examples.get(name, 0),
            'own_class_scored': len(own),
            'novel_scored': len(novel),
            'auroc': auroc(novel, own),
            'judged': examples.get(name, 0) >= MIN_EXAMPLES and len(novel) >= MIN_NOVEL,
        }
    return out


def best_pair_cosine(vectors: Floats, a: Ints, b: Ints) -> float:
    """Largest cosine between a member of cluster ``a`` and one of ``b`` (row index arrays)."""
    return float((vectors[a] @ vectors[b].T).max())


def suggestions(
    vectors: Floats, part: Ints, keys: Any, flagged_clusters: Sequence[int]
) -> list[tuple[float, int, int]]:
    """Cross-key merge suggestions between flagged clusters, best member pair first."""
    members: dict[int, Ints] = {c: np.nonzero(part == c)[0] for c in flagged_clusters}
    key_of = {c: str(np.asarray(keys)[m[0]]) for c, m in members.items()}
    ids = sorted(members)
    pairs = [
        (best_pair_cosine(vectors, members[a], members[b]), a, b)
        for i, a in enumerate(ids)
        for b in ids[i + 1 :]
        if key_of[a] != key_of[b]
    ]
    return sorted(pairs, reverse=True)


def majority_open_label(
    part: Ints, cluster: int, truth: Any, judged: Mask, rows: Ints, key_set: set[str]
) -> str | None:
    """The most common open-set label among the judged labelled members, or None."""
    counts: dict[str, int] = defaultdict(int)
    for j in np.nonzero(judged & (part[rows] == cluster))[0]:
        if str(truth[j]) not in key_set:
            counts[str(truth[j])] += 1
    return max(counts, key=lambda c: (counts[c], c)) if counts else None


def suggestion_correct(
    pair: tuple[float, int, int],
    part: Ints,
    truth: Any,
    judged: Mask,
    rows: Ints,
    key_set: set[str],
) -> bool | None:
    """True if both clusters' majority open-set labels agree; None if either has none."""
    la = majority_open_label(part, pair[1], truth, judged, rows, key_set)
    lb = majority_open_label(part, pair[2], truth, judged, rows, key_set)
    return None if la is None or lb is None else la == lb


def threshold_for_precision(
    ranked: Sequence[tuple[float, bool]], bar: float, min_count: int = MIN_SUGGESTIONS
) -> float | None:
    """The lowest score at which suggestions at or above it have precision >= ``bar``.

    ``ranked`` is ``(score, correct)`` best first; at least ``min_count`` must qualify.
    """
    best, hits = None, 0
    for i, (score, ok) in enumerate(ranked, 1):
        hits += ok
        if i >= min_count and hits / i >= bar:
            best = score
    return best


def merge_clusters(part: Ints, pairs: Sequence[tuple[float, int, int]]) -> Ints:
    """Partition with each suggested pair of clusters merged (union of connected clusters)."""
    parent = {int(c): int(c) for c in np.unique(part)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, a, b in pairs:
        parent[find(a)] = find(b)
    return np.array([find(int(c)) for c in part], dtype=np.int64)


def open_set_pair_recall(truth: Any, part: Ints, mask: Mask, key_set: set[str]) -> float | None:
    """Pair recall over crops whose label is open-set (same-label pairs kept together)."""
    t = np.asarray(truth)
    sel = mask & ~np.isin(t, list(key_set))
    if sel.sum() < 2:
        return None
    return met.pairwise_scores(t[sel], part[sel])['pair_recall']  # type: ignore[no-any-return]


def source_groups(b: Bundle) -> list[str]:
    """The source image of every crop: the label group where known, else the image file name."""
    known = {image_name(b.meta[i]): g for i, g in zip(b.labelled, b.group, strict=True)}
    return [known.get(image_name(m), image_name(m)) for m in b.meta]


def _suggestion_report(
    pairs: Sequence[tuple[float, int, int]],
    part: Ints,
    truth: Any,
    fit: Mask,
    scored: Mask,
    rows: Ints,
    key_set: set[str],
) -> dict[str, Any]:
    """Precision of the cross-key suggestions: threshold chosen on ``fit``, judged on ``scored``."""

    def ranked(judged: Mask) -> list[tuple[float, bool]]:
        found = [(p[0], suggestion_correct(p, part, truth, judged, rows, key_set)) for p in pairs]
        return [(s, ok) for s, ok in found if ok is not None]

    fit_ranked, scored_ranked = ranked(fit), ranked(scored)
    thr = threshold_for_precision(fit_ranked, SUGGESTION_PRECISION_BAR)
    chosen = [(s, ok) for s, ok in scored_ranked if thr is not None and s >= thr]
    return {
        'suggestions': len(pairs),
        'evaluable_fit': len(fit_ranked),
        'evaluable_scored': len(scored_ranked),
        'threshold_from_fit': thr,
        'scored_at_threshold': len(chosen),
        'scored_precision_at_threshold': sum(ok for _, ok in chosen) / len(chosen)
        if chosen
        else None,
        'scored_precision_at_k': {
            str(k): sum(ok for _, ok in scored_ranked[:k]) / k
            for k in (5, 10, 20)
            if len(scored_ranked) >= k
        },
    }


def run_fold(b: Bundle, keys: Any, part: Ints, fit: Mask, scored: Mask) -> dict[str, Any]:
    """Novelty AUROC, flag precision and cross-key suggestions for one fold.

    ``fit`` and ``scored`` are masks over the labelled crops; ``part`` covers every crop.
    """
    lab, k_all = b.labelled, np.asarray(keys)
    truth = np.array([normalize_class_name(str(t)) for t in b.truth])
    key_set = set(k_all.tolist())
    keys_lab = k_all[lab]
    fit_idx, scored_idx = np.nonzero(fit)[0], np.nonzero(scored)[0]
    own = fit_idx[truth[fit_idx] == keys_lab[fit_idx]]
    crop = crop_novelty(b.vectors, k_all, source_groups(b), lab[own], keys_lab[own])
    clus = cluster_novelty(part, crop)
    crop_lab, clus_lab = crop[lab], clus[lab]
    novel = ~np.isin(truth, list(key_set))
    examples = {n: int((keys_lab[own] == n).sum()) for n in set(keys_lab[own].tolist())}
    q, table = choose_quantile(keys_lab, clus_lab, own, fit_idx, novel[fit_idx])
    out: dict[str, Any] = {
        'fit_labelled': int(fit.sum()),
        'scored_labelled': int(scored.sum()),
        'open_set_crops_scored': int(novel[scored_idx].sum()),
        'per_key_auroc_cluster_score': per_key_auroc(
            keys_lab, clus_lab, scored_idx, truth, key_set, examples
        ),
        'per_key_auroc_crop_score': per_key_auroc(
            keys_lab, crop_lab, scored_idx, truth, key_set, examples
        ),
        'quantile_table_fit_half': table,
        'chosen_quantile': q,
    }
    judged = [v for v in out['per_key_auroc_cluster_score'].values() if v['judged']]
    worst = min((v['auroc'] for v in judged), default=None)
    out['auroc_bar'] = {
        'judged_keys': len(judged),
        'min_auroc': worst,
        'bar': AUROC_BAR,
        'pass': None if worst is None else worst >= AUROC_BAR,
    }
    out['flag_quality_scored_by_quantile'] = {
        str(level): flag_quality(
            flagged(keys_lab, clus_lab, own_class_thresholds(keys_lab, clus_lab, own, level))[
                scored_idx
            ],
            novel[scored_idx],
        )
        for level in QUANTILES
    }
    # no quantile reaching the precision bar: the highest one is evaluated, labelled as such
    level = QUANTILES[-1] if q is None else q
    out['suggestion_basis'] = (
        'calibrated_quantile' if q is not None else 'exploratory_highest_quantile'
    )
    th = own_class_thresholds(keys_lab, clus_lab, own, level)
    out['flag_quality_scored'] = flag_quality(
        flagged(keys_lab, clus_lab, th)[scored_idx], novel[scored_idx]
    )
    flag_all = flagged(k_all, clus, th)
    clusters = [int(c) for c in np.unique(part[flag_all])]
    pairs = suggestions(b.vectors, part, k_all, clusters)
    sug = _suggestion_report(pairs, part, truth, fit, scored, lab, key_set)
    out['suggestions'] = sug
    if sug['threshold_from_fit'] is not None:
        merged = merge_clusters(part, [p for p in pairs if p[0] >= sug['threshold_from_fit']])
        out['merge_effect_scored'] = {
            'open_set_pair_recall_before': open_set_pair_recall(truth, part[lab], scored, key_set),
            'open_set_pair_recall_after': open_set_pair_recall(truth, merged[lab], scored, key_set),
            'bcubed_precision_before': met.bcubed_scores(truth[scored], part[lab][scored])[
                'bcubed_precision'
            ],
            'bcubed_precision_after': met.bcubed_scores(truth[scored], merged[lab][scored])[
                'bcubed_precision'
            ],
        }
    return out
