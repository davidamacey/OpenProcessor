"""Average-linkage clustering inside the detector class, and the scoring views around it.

The partition key is the crop's detector class name (``dataset.detector_keys``); each key gets
its own average-linkage dendrogram and one cosine-distance cut is applied to all of them, so
a crop never joins another key's cluster. The scoring views repeat every figure on the
subsets that would otherwise hide a weak spot: the non-person crops (the person class carries
most same-label pairs on COCO-like sets), each stratum, and the trivial baselines that need no
clustering at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench import cluster_eval_metrics as met
from scripts.research.cluster_diag import diagnostics as dg, partitions as pt


if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    Ints = NDArray[np.integer[Any]]
    Floats = NDArray[np.floating[Any]]
    Mask = NDArray[np.bool_]

PERSON = 'person'
QUANTILE_GRID_POINTS = 60


def quantile_cut_grid(vectors: Floats) -> list[float]:
    """Cut heights at 60 quantiles of the global average-linkage merge heights (the E5 grid)."""
    z = dg.linkage_of(vectors @ vectors.T, 'average')
    return [float(t) for t in np.quantile(z[:, 2], np.linspace(0.5, 0.995, QUANTILE_GRID_POINTS))]


def uniform_cut_grid(lo: float, hi: float, step: float) -> list[float]:
    """Evenly spaced cut heights, ``hi`` included; fine enough to compare cuts to 0.01."""
    return [float(t) for t in np.arange(lo, hi + step / 2, step)]


class WithinKeyClusterer:
    """One average-linkage dendrogram per key, cut at any height without refitting."""

    def __init__(self, vectors: Floats, keys: Any) -> None:
        self.n = len(vectors)
        self._linkages = pt.group_linkages(keys, vectors @ vectors.T)

    def partition(self, cut: float) -> Ints:
        return pt.cut_groups(self.n, self._linkages, float(cut))


def key_only_partition(keys: Any) -> Ints:
    """The trivial baseline: one cluster per key, no clustering inside it."""
    return np.unique(np.asarray(keys).astype(str), return_inverse=True)[1].astype(np.int64)


def trivial_partitions(keys: Any) -> dict[str, Ints]:
    """Partitions that need no clustering: the key alone, every crop alone, everything together."""
    n = len(keys)
    return {
        'detector_class_only': key_only_partition(keys),
        'all_singletons': np.arange(n, dtype=np.int64),
        'one_cluster': np.zeros(n, dtype=np.int64),
    }


def score_block(
    truth: Any, part: Ints, mask: Mask, *, with_ami: bool = False
) -> dict[str, float | int | None]:
    """Pair and BCubed scores of the masked labelled crops (AMI on request)."""
    t, p = np.asarray(truth)[mask], part[mask]
    pair = met.pairwise_scores(t, p)
    bc = met.bcubed_scores(t, p)
    out: dict[str, float | int | None] = {
        'n': int(mask.sum()),
        'clusters': pair['clusters'],
        'pair_precision': pair['pair_precision'],
        'pair_recall': pair['pair_recall'],
        'bcubed_precision': bc['bcubed_precision'],
        'bcubed_recall': bc['bcubed_recall'],
        'bcubed_f': bc['bcubed_f'],
    }
    if with_ami:
        out['ami'] = met.adjusted_mutual_info(t, p)
    return out


def person_pair_share(truth: Any, mask: Mask) -> float | None:
    """Share of the masked same-label pairs that are person pairs."""
    t = np.asarray(truth)[mask]
    _, counts = np.unique(t, return_counts=True)
    total = int((counts * (counts - 1) // 2).sum())
    n_person = int((t == PERSON).sum())
    return n_person * (n_person - 1) // 2 / total if total else None


def views(
    truth: Any, part: Ints, scored: Mask, strata: Mapping[str, Mask]
) -> dict[str, dict[str, float | int | None]]:
    """Scores on all scored crops, the non-person and person crops, and each stratum."""
    t = np.asarray(truth)
    out = {
        'all': score_block(t, part, scored, with_ami=True),
        'non_person': score_block(t, part, scored & (t != PERSON)),
        'person': score_block(t, part, scored & (t == PERSON)),
    }
    for name, mask in strata.items():
        if (scored & mask).sum() > 1:
            out[f'stratum:{name}'] = score_block(t, part, scored & mask)
    return out
