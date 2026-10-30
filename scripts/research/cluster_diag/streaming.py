"""Late arrivals: assign a new crop to an existing cluster of its own detector class, or abstain.

A seeded share of the source images is held back. The rest are clustered inside each detector
class at the product cut. Each held-back crop goes to the nearest cluster of its own key by
centroid, by best member or by mean member distance, and abstains when that distance is above
the cut or the margin to the second-nearest cluster is too small. The margin is chosen on the
arrivals of one half of the images and the accepted assignments are scored on the other half.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench.parity_metrics import adjusted_rand_index
from scripts.research.cluster_diag import class_prior as cp
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray

    Floats = NDArray[np.floating[Any]]
    Ints = NDArray[np.integer[Any]]
    Mask = NDArray[np.bool_]

METHODS = ('centroid', 'best_member', 'mean_member')
MARGINS = (0.0, 0.01, 0.02, 0.05, 0.1)
PRECISION_BAR = 0.95
ARRIVAL_SHARE = 0.2
SEEDS = (20261029, 1, 2, 3, 4)


def late_groups(groups: Iterable[str], share: float, seed: int) -> set[str]:
    """The seeded ``share`` of the distinct groups (images) that arrive late.

    The ranking has its own namespace, so it is independent of the calibration/held-out split
    that ranks the same images by the same seed.
    """
    ranked = sorted(
        set(groups), key=lambda g: hashlib.sha256(f'late:{seed}:{g}'.encode()).hexdigest()
    )
    return set(ranked[: round(len(ranked) * share)])


def cluster_distances(x: Floats, members: Floats, method: str) -> float:
    """Distance from one unit vector to a cluster's unit member vectors."""
    sims = members @ x
    if method == 'best_member':
        return float(1.0 - sims.max())
    if method == 'mean_member':
        return float(1.0 - sims.mean())
    centroid = members.mean(0)
    return float(1.0 - x @ centroid / np.linalg.norm(centroid))


def assign(
    x: Floats, clusters: dict[int, Floats], method: str, cut: float, margin: float
) -> int | None:
    """The nearest cluster of ``clusters`` or None: abstain above ``cut`` or inside ``margin``."""
    if not clusters:
        return None
    dist = sorted((cluster_distances(x, m, method), cid) for cid, m in clusters.items())
    if dist[0][0] > cut:
        return None
    if len(dist) > 1 and dist[1][0] - dist[0][0] < margin:
        return None
    return dist[0][1]


def key_clusters(
    vectors: Floats, keys: Any, part: Ints, base: Mask
) -> dict[str, dict[int, Floats]]:
    """``{key: {cluster id: member vectors}}`` over the base crops."""
    members: dict[str, dict[int, list[Floats]]] = {}
    for row in np.nonzero(base)[0]:
        members.setdefault(str(keys[row]), {}).setdefault(int(part[row]), []).append(vectors[row])
    return {k: {c: np.stack(v) for c, v in d.items()} for k, d in members.items()}


def plurality(labels: Sequence[str]) -> str | None:
    """Most common label (ties broken alphabetically), None if there are none."""
    if not labels:
        return None
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(sorted(counts), key=lambda c: counts[c])


def score_arrivals(
    assigned: dict[int, int | None],
    truth_of: dict[int, str],
    cluster_labels: dict[int, str | None],
    nameable: set[int] | None = None,
) -> dict[str, Any]:
    """Accepted share and the share of accepted arrivals matching their cluster's plurality.

    ``nameable`` are the arrivals whose label is a class the detector can output; their
    precision is reported on its own, since an open-set arrival can never match a cluster that
    the detector's class decided.
    """
    total = len(assigned)
    accepted = {r: c for r, c in assigned.items() if c is not None}
    judged = [(r, c) for r, c in accepted.items() if cluster_labels.get(c) is not None]
    hits = sum(truth_of[r] == cluster_labels[c] for r, c in judged)
    known = [(r, c) for r, c in judged if nameable is None or r in nameable]
    known_hits = sum(truth_of[r] == cluster_labels[c] for r, c in known)
    return {
        'judged_nameable': len(known),
        'label_precision_nameable': known_hits / len(known) if known else None,
        'arrivals': total,
        'accepted': len(accepted),
        'abstain_share': 1 - len(accepted) / total if total else None,
        'judged': len(judged),
        'label_precision': hits / len(judged) if judged else None,
    }


def choose_margin(
    table: dict[float, dict[str, Any]], bar: float = PRECISION_BAR, min_judged: int = 10
) -> float | None:
    """The smallest margin whose accepted precision reaches ``bar`` on enough judged arrivals."""
    for margin in sorted(table):
        row = table[margin]
        if row['judged'] >= min_judged and (row['label_precision'] or 0.0) >= bar:
            return margin
    return None


def incremental_vs_refit(
    base_part: Ints, assignment: dict[int, int | None], refit: Ints, base: Mask
) -> dict[str, float]:
    """ARI of the incremental partition against a full refit (abstained arrivals are singletons)."""
    inc = np.array(base_part, dtype=np.int64)
    next_id = int(inc.max()) + 1
    for row in sorted(assignment):
        cid = assignment[row]
        inc[row] = next_id if cid is None else cid
        next_id += cid is None
    return {
        'ari_all_crops': float(adjusted_rand_index(inc, refit)),
        'ari_base_crops': float(adjusted_rand_index(inc[base], refit[base])),
    }


def run_streaming(
    vectors: Floats,
    keys: Any,
    groups_all: Sequence[str],
    labelled: Ints,
    truth: Any,
    first_half_groups: set[str],
    cut: float,
    seed: int,
) -> dict[str, Any]:
    """One seeded late-arrival run: per method, margin chosen on one image half, scored on the other."""
    late = late_groups(groups_all, ARRIVAL_SHARE, seed)
    is_late = np.array([g in late for g in groups_all])
    base = ~is_late
    k_arr = np.asarray(keys)
    base_rows = np.nonzero(base)[0]
    base_part = np.full(len(vectors), -1, dtype=np.int64)
    base_part[base_rows] = cp.WithinKeyClusterer(vectors[base_rows], k_arr[base_rows]).partition(
        cut
    )
    refit = cp.WithinKeyClusterer(vectors, k_arr).partition(cut)
    truth_of = {int(r): normalize_class_name(str(t)) for r, t in zip(labelled, truth, strict=True)}
    key_set = set(k_arr.tolist())
    nameable = {r for r, t in truth_of.items() if t in key_set}
    clusters = key_clusters(vectors, k_arr, base_part, base)
    cluster_label = {
        c: plurality(
            [truth_of[int(r)] for r in np.nonzero(base & (base_part == c))[0] if int(r) in truth_of]
        )
        for c in {int(x) for x in base_part[base_rows]}
    }
    arrivals = [int(r) for r in np.nonzero(is_late)[0]]
    in_first = {r: groups_all[r] in first_half_groups for r in arrivals}
    out: dict[str, Any] = {
        'seed': seed,
        'late_images': len(late),
        'arrivals': len(arrivals),
        'methods': {},
    }
    for method in METHODS:
        per_margin: dict[float, dict[str, Any]] = {}
        for margin in MARGINS:
            asg = {
                r: assign(vectors[r], clusters.get(str(k_arr[r]), {}), method, cut, margin)
                for r in arrivals
            }
            halves = {}
            for half, flag in (('first', True), ('second', False)):
                sub = {r: c for r, c in asg.items() if in_first[r] == flag}
                lab_sub = {r: c for r, c in sub.items() if r in truth_of}
                halves[half] = score_arrivals(lab_sub, truth_of, cluster_label, nameable) | {
                    'abstain_share_all_arrivals': sum(c is None for c in sub.values()) / len(sub)
                    if sub
                    else None
                }
            per_margin[margin] = {'halves': halves, 'assignment': asg}
        folds = {}
        for name, fit_half, test_half in (
            ('fold1_first_to_second', 'first', 'second'),
            ('fold2_second_to_first', 'second', 'first'),
        ):
            chosen = choose_margin({m: v['halves'][fit_half] for m, v in per_margin.items()})
            row: dict[str, Any] = {'margin': chosen}
            if chosen is not None:
                at = per_margin[chosen]
                row['scored'] = at['halves'][test_half]
                row['incremental_vs_refit'] = incremental_vs_refit(
                    base_part, at['assignment'], refit, base
                )
            folds[name] = row
        out['methods'][method] = {
            'margin_table': {str(m): v['halves'] for m, v in per_margin.items()},
            'folds': folds,
        }
    return out
