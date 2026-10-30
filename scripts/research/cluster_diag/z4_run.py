"""Experiment Z4 (decision note section 0A.2): the class-level root choice for the auto-label loop.

    z4_run.py --bundle STEM --labels L --coco-gt G --split-manifest S --out OUT.json

Arms, each a partition of the labelled crops of the held-out half used as the loop's root nodes:
``key`` (one root per detector key), ``key_guard`` (the same plus the label-free hold-back guard),
``within_key`` (average linkage inside each key at the cut the section-6 operating-point rule
chooses on the calibration half) and ``ivf_loop`` (the production IVF refit). Each runs the loop's
recursion (``loop_sim``). ``ivf_gate`` is IVF asked whole when it fails (the Z2 convention). The
bars B-a to B-c (``z4_bars``) are read once, fold 1, with the clean oracle labeler. The noisy and
the registry-closed rows are not barred.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench import scorecard
from scripts.datasets.vehicle_eval_selection import OPENSET
from scripts.research.cluster_diag import (
    acceptance as acc,
    class_prior as cp,
    gate_sim as gs,
    loop_sim as ls,
    partitions as pt,
    y_acceptance as ya,
    z4_bars as zb,
)
from scripts.research.cluster_diag.dataset import detector_keys, load_bundle
from scripts.research.cluster_diag.run import calibration_mask
from scripts.research.cluster_diag.z_run import SIM_SEEDS


def composition(truth: Any, keys: Any, pool: Any) -> dict[str, Any]:
    """Open-set crops in the whole labelled set and in the scored pool, and the detector keys
    they sit under (the clusters that can expose them as a minority)."""
    t, k = np.asarray(truth).astype(str), np.asarray(keys).astype(str)
    opened = np.isin(t, sorted(OPENSET))
    return {
        'labelled_open_set_all': int(opened.sum()),
        'labelled_open_set_pool': int((opened & pool).sum()),
        'pool_items': int(pool.sum()),
        'open_set_by_class_pool': dict(sorted(Counter(t[opened & pool].tolist()).items())),
        'open_set_by_detector_key_pool': dict(sorted(Counter(k[opened & pool].tolist()).items())),
        'pool_by_detector_key': dict(sorted(Counter(k[pool].tolist()).items())),
    }


def key_exposure(truth: Any, keys: Any, pool: Any) -> list[dict[str, Any]]:
    """Per detector key of the pool: size, the plurality class's share and the open-set count."""
    t, k = np.asarray(truth).astype(str), np.asarray(keys).astype(str)
    rows = []
    for key in sorted(set(k[pool].tolist())):
        member = pool & (k == key)
        classes = Counter(t[member].tolist())
        top, top_n = classes.most_common(1)[0]
        rows.append(
            {
                'key': key,
                'items': int(member.sum()),
                'plurality_class': top,
                'plurality_share': top_n / int(member.sum()),
                'open_set_items': int(np.isin(t[member], sorted(OPENSET)).sum()),
            }
        )
    return rows


def run_z4(args: argparse.Namespace) -> dict[str, Any]:
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    keys = detector_keys(b.meta)
    first = calibration_mask(b, args.split_manifest)
    pool = ~first
    arm = ya.global_cut_arm(b, keys)
    op = acc.operating_point(arm.candidates, b.truth, first)
    truth = np.asarray(b.truth).astype(str)
    out: dict[str, Any] = {
        'registered': {
            'u2_bar': zb.U2_BAR,
            'ivf_factor': zb.IVF_FACTOR,
            'open_bar': zb.OPEN_BAR,
            'min_open_propagated': zb.MIN_OPEN_PROPAGATED,
            'sim_seeds': list(SIM_SEEDS),
            'noise': gs.NOISE,
            'gate': {'theta': gs.THETA, 'alpha': gs.ALPHA, 'max_dissent': gs.MAX_DISSENT},
            'recursion': {
                'branching': ls.BRANCHING,
                'max_depth': ls.MAX_DEPTH,
                'min_node': ls.MIN_NODE,
                'max_refine': ls.MAX_REFINE,
            },
            'guard_percentile': ls.GUARD_PERCENTILE,
            'open_set_classes': sorted(OPENSET),
        },
        'crops': len(b.vectors),
        'labelled': len(truth),
        'calibration_labelled': int(first.sum()),
        'heldout_labelled': int(pool.sum()),
        'operating_point': op,
        'composition': composition(truth, keys[b.labelled], pool),
        'key_exposure_heldout': key_exposure(truth, keys[b.labelled], pool),
    }
    pool_vectors = np.ascontiguousarray(b.vectors[b.labelled][pool])
    t = truth[pool].tolist()
    parts: dict[str, Any] = {
        'key': cp.key_only_partition(np.asarray(keys)[b.labelled])[pool],
        'ivf_loop': pt.ivf_current(b.vectors)[b.labelled][pool],
    }
    parts['key_guard'] = parts['key']
    if op is not None:
        parts['within_key'] = np.asarray(arm.full(float(op['parameter']))[b.labelled])[pool]
    splitter = ls.AhcSplitter(pool_vectors)
    closed = frozenset(set(t) - OPENSET)

    def loop(name: str, noise: float = 0.0, registry: frozenset[str] | None = None) -> dict:
        return ls.seeded_loop(
            parts[name],
            t,
            pool_vectors,
            splitter,
            seeds=SIM_SEEDS,
            open_set=OPENSET,
            guard=name == 'key_guard',
            noise=noise,
            registry=registry,
        )

    arms = {name: loop(name) for name in (*zb.ARMS, 'ivf_loop') if name in parts}
    ivf_gate = gs.seeded_runs(parts['ivf_loop'], t, seeds=SIM_SEEDS, open_set=OPENSET)
    out['arms_clean'] = arms
    out['ivf_asked_whole'] = ivf_gate
    ivf_u1 = zb.ivf_reference(arms['ivf_loop'], ivf_gate)
    rows = {name: zb.arm_rows(arms[name], ivf_u1) for name in zb.ARMS if name in arms}
    verdicts = {name: zb.verdict(r) for name, r in rows.items()}
    if 'within_key' not in arms:
        rows['within_key'] = [{'id': 'all', 'result': zb.FAIL, 'note': 'no operating point'}]
        verdicts['within_key'] = zb.FAIL
    out['bars'] = {
        'ivf_u1_reference': ivf_u1,
        'rows': rows,
        'verdicts': verdicts,
        'adoption': zb.adoption(
            verdicts, {n: (arms[n]['u1']['mean'] if n in arms else None) for n in zb.ARMS}
        ),
        'overall': zb.overall(verdicts),
    }
    out['noisy_0.10_not_barred'] = {
        n: loop(n, noise=gs.NOISE) for n in (*zb.ARMS, 'ivf_loop') if n in parts
    }
    out['registry_closed_not_barred'] = {
        'registry': sorted(closed),
        **{n: loop(n, registry=closed) for n in zb.ARMS if n in parts},
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for opt in ('bundle', 'labels', 'coco-gt', 'split-manifest', 'out'):
        ap.add_argument(f'--{opt}', type=Path, required=True)
    args = ap.parse_args(argv)
    start = os.getloadavg()
    result = {
        'command': 'z4',
        'git_sha': scorecard.current_git_sha(),
        'load_avg_start': start,
        **run_z4(args),
        'load_avg_end': os.getloadavg(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
