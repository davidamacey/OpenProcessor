"""Write one arm's partition of an exported project so ``cluster_eval.py score`` can score it.

    export_partition.py --bundle STEM --arm ahc_average --param 0.41 --out DIR/partition.jsonl

``--param`` is the AHC cut height or the Leiden resolution fraction, as printed in the ``e1``
output (``parameter``); the IVF, ECG, 2b and detector-class arms take none. ``ahc_average_within_class``
cuts one average-linkage dendrogram per detector class at ``--param``; ``detector_class`` is that
class alone, with no clustering (the trivial baseline). Rows are ``item_id``, ``cluster_id``
(``null`` for an unassigned crop), over every crop of the export, so the scorer applies the
labels and any held-out split itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.cluster.hierarchy import fcluster


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.research.cluster_diag import class_prior as cp, diagnostics as dg, partitions as pt
from scripts.research.cluster_diag.dataset import detector_keys, read_jsonl


ARMS = (
    'ivf',
    'ahc_average',
    'ahc_complete',
    'leiden',
    'ecg',
    'kmeans2b',
    'detector_class',
    'ahc_average_within_class',
)


def partition_of(arm: str, vectors: Any, param: float | None, keys: Any = None) -> Any:
    if arm == 'ivf':
        return pt.ivf_current(vectors)
    if arm == 'kmeans2b':
        return pt.region_kmeans(vectors)
    if arm == 'detector_class':
        return cp.key_only_partition(keys)
    if param is None and arm not in ('ecg',):
        raise SystemExit(f'--param is required for {arm}')
    if arm == 'ahc_average_within_class':
        return cp.WithinKeyClusterer(vectors, keys).partition(float(param or 0.0))
    sim = vectors @ vectors.T
    if arm == 'ecg':
        return pt.ecg_partition(sim)
    if arm == 'leiden':
        return dg.leiden_cpm_partition(sim, 30, float(param or 0.0))
    method = arm.removeprefix('ahc_')
    return (
        fcluster(dg.linkage_of(sim, method), float(param or 0.0), 'distance').astype(np.int64) - 1
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--param', type=float, default=None)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    vectors = dg.l2_normalize(np.load(args.bundle.with_suffix('.npy')))
    meta = read_jsonl(args.bundle.with_suffix('.meta.jsonl'))
    ids = [m['crop_id'] for m in meta]
    part = partition_of(args.arm, vectors, args.param, detector_keys(meta))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        ''.join(
            json.dumps({'item_id': i, 'cluster_id': None if c < 0 else int(c)}) + '\n'
            for i, c in zip(ids, part, strict=True)
        )
    )
    print(
        f'{args.arm}: {len(ids)} crops, {len({int(c) for c in part if c >= 0})} clusters -> {args.out}'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
