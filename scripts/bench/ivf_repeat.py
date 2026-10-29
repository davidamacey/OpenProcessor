"""Same-seed repeatability of the production IVF pass on one project's stored vectors (#341).

    ivf_repeat.py --opensearch-url http://127.0.0.1:9200 --index op_prj_<slug>__items \\
        --out-dir artifacts_local/cluster_eval/<slug>/repeat [--runs 2] [--shuffle-seed 7]

Reads the project's stored crop vectors (read only), then fits ``IVFMethod`` (the clusterer the
refresh worker runs, seed 42, centroids not persisted) ``--runs`` times on the same vectors in
the same order, and once more on a seeded permutation of the rows. Each fit is written as a
``partition_*.jsonl`` that ``cluster_eval.py score --second-partition`` can compare. The summary
JSON says whether the same-order runs are identical and how far the permuted run is from them
(adjusted Rand index), because the k-means initialisation and the training sample depend on row
order. Run it on a throw-away project or a copy: it never writes to the index.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.export_project_embeddings import export
from scripts.bench.parity_metrics import adjusted_rand_index
from src.services.curation.clustering.methods import ivf_store
from src.services.curation.clustering.methods.ivf import IVFMethod


def fit_once(vectors: np.ndarray) -> np.ndarray:
    """Cluster labels from the production IVF method on the CPU.

    The method saves its centroids and assigns against the saved copy, so they go to a temporary
    project state directory that is removed afterwards; no project directory is touched.
    """
    with tempfile.TemporaryDirectory() as state:
        cfg = SimpleNamespace(project_state_dir=Path(state))
        with mock.patch.object(ivf_store, 'get_curation_config', return_value=cfg):
            result = asyncio.run(
                IVFMethod().fit_predict(vectors, backend_info=SimpleNamespace(name='cpu'))  # type: ignore[arg-type]
            )
    return np.asarray(result.labels)


def repeat(vectors: np.ndarray, runs: int, shuffle_seed: int) -> dict[str, Any]:
    """Labels of ``runs`` same-order fits and of one fit on a permuted copy (back in row order)."""
    same = []
    for _ in range(runs):
        started = time.perf_counter()
        same.append((fit_once(vectors), time.perf_counter() - started))
    order = np.random.default_rng(shuffle_seed).permutation(len(vectors))
    permuted = np.empty(len(vectors), dtype=np.int64)
    permuted[order] = fit_once(vectors[order])
    first = same[0][0]
    return {
        'labels': [lab for lab, _ in same],
        'permuted_labels': permuted,
        'fit_seconds': [round(sec, 3) for _, sec in same],
        'same_order_identical': all(np.array_equal(first, lab) for lab, _ in same[1:]),
        'same_order_ari': [float(adjusted_rand_index(first, lab)) for lab, _ in same[1:]],
        'permuted_ari': float(adjusted_rand_index(first, permuted)),
        'permuted_identical': bool(np.array_equal(first, permuted)),
    }


def write_partition(path: Path, ids: Sequence[str], labels: np.ndarray) -> None:
    path.write_text(
        ''.join(
            json.dumps({'item_id': i, 'cluster_id': int(c)}) + '\n'
            for i, c in zip(ids, labels, strict=True)
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--opensearch-url', required=True)
    ap.add_argument('--index', required=True, help='the project items index (read only)')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--runs', type=int, default=2)
    ap.add_argument('--shuffle-seed', type=int, default=7)
    ap.add_argument('--page', type=int, default=2000)
    args = ap.parse_args(argv)
    vectors, _, ids = export(args.opensearch_url, args.index, args.page)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    out = repeat(vectors, args.runs, args.shuffle_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n, lab in enumerate(out.pop('labels'), 1):
        write_partition(args.out_dir / f'partition_run{n}.jsonl', ids, lab)
    write_partition(args.out_dir / 'partition_permuted.jsonl', ids, out.pop('permuted_labels'))
    summary = {'index': args.index, 'vectors': len(ids), 'dim': int(vectors.shape[1]), **out}
    (args.out_dir / 'repeat.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
