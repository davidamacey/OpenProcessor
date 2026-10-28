"""Score a cluster partition against independent labels and emit JSON.

    cluster_eval.py score --labels labels.jsonl --partition partition.jsonl --out score.json \\
        [--second-partition rerun.jsonl] [--reference current.jsonl] \\
        [--runtime-s 123.4] [--peak-rss-mb 2048]

``labels.jsonl`` has one object per line: ``item_id``, ``label`` (the ground-truth class name),
optionally ``stratum`` (a hard-case group reported on its own) and ``protected`` (true for a
human-validated, import-validated, locked or holdout item). ``partition.jsonl`` has ``item_id``
and ``cluster_id`` (null or absent for an unassigned item). Any producer works: the current
pipeline, a candidate algorithm or a rerun. A labelled item missing from the partition counts as
unassigned; partition items without a label are counted and ignored.

``--second-partition`` adds the stability of two runs. ``--reference`` adds how many protected
items the partition moves relative to the current one. ``--runtime-s`` and ``--peak-rss-mb`` are
the producer's own cost, recorded as given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench import cluster_eval_metrics as metrics


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise SystemExit(f'{path}:{n}: not JSON ({exc})') from exc
            if not isinstance(row, dict) or 'item_id' not in row:
                raise SystemExit(f'{path}:{n}: expected an object with item_id')
            rows.append(row)
    return rows


def read_labels(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Item ids in file order and aligned ``truth``, ``stratum`` and ``protected`` arrays."""
    rows = _jsonl(path)
    ids = [str(r['item_id']) for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f'{path}: duplicate item_id')
    if any('label' not in r for r in rows):
        raise SystemExit(f'{path}: every row needs a label')
    return ids, {
        'truth': np.array([str(r['label']) for r in rows]),
        'stratum': np.array([str(r.get('stratum', 'all')) for r in rows]),
        'protected': np.array([bool(r.get('protected', False)) for r in rows], dtype=bool),
    }


def read_partition(path: Path, ids: Sequence[str]) -> tuple[np.ndarray, int]:
    """Cluster codes aligned to ``ids`` (unassigned = -1) and the count of unlabelled rows."""
    code: dict[str, int] = {}
    by_item: dict[str, int] = {}
    for r in _jsonl(path):
        item = str(r['item_id'])
        if item in by_item:
            raise SystemExit(f'{path}: duplicate item_id {item}')
        cid = r.get('cluster_id')
        by_item[item] = metrics.UNASSIGNED if cid is None else code.setdefault(str(cid), len(code))
    known = set(ids)
    extra = sum(1 for item in by_item if item not in known)
    return np.array([by_item.get(i, metrics.UNASSIGNED) for i in ids], dtype=np.int64), extra


def score(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    ids, lab = read_labels(Path(args.labels))
    cluster, extra = read_partition(Path(args.partition), ids)
    out: dict[str, Any] = {
        'partition': args.partition,
        'labels': args.labels,
        'partition_rows_without_label': extra,
        'classes': len(np.unique(lab['truth'])),
        **metrics.pairwise_scores(lab['truth'], cluster),
    }
    if len(np.unique(lab['stratum'])) > 1:
        out['strata'] = metrics.stratified_scores(lab['truth'], cluster, lab['stratum'])
    if args.second_partition:
        second, _ = read_partition(Path(args.second_partition), ids)
        out['stability'] = metrics.assignment_stability(cluster, second)
    if args.reference:
        ref, _ = read_partition(Path(args.reference), ids)
        out['protected_items'] = int(lab['protected'].sum())
        out['protected_items_moved'] = metrics.moved_items(ref, cluster, lab['protected'])
    out['producer_runtime_s'] = args.runtime_s
    out['producer_peak_rss_mb'] = args.peak_rss_mb
    out['scoring_s'] = time.perf_counter() - started
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score', help='score one partition and write JSON')
    s.add_argument('--labels', required=True)
    s.add_argument('--partition', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--second-partition')
    s.add_argument('--reference')
    s.add_argument('--runtime-s', type=float)
    s.add_argument('--peak-rss-mb', type=float)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = score(args)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
