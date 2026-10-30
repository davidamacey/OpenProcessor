"""Audit the items a candidate partition moves relative to the current one, against labels.

    move_audit.py --labels labels.jsonl --current cur.jsonl --candidate cand.jsonl

A reference (current) cluster is mapped to the candidate cluster holding most of its members (the
successor); an item "moves" when it is not in its cluster's successor. The successor mapping makes
a merge of whole clusters free, so only items that cross a boundary count. For each moved item the
class of its old cluster (majority label) and of its new cluster are compared with its own label:
``fixed`` (wrong before, right after), ``broken`` (right before, wrong after), ``both_right``,
``both_wrong``. Items without a label or a cluster on either side are skipped and counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path, key: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    with path.open() as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                v = row.get(key)
                out[str(row['item_id'])] = None if v is None else str(v)
    return out


def _majority(cluster: np.ndarray, label: np.ndarray) -> dict[int, int]:
    pairs, counts = np.unique(np.stack([cluster, label]), axis=1, return_counts=True)
    best: dict[int, tuple[int, int]] = {}
    for (c, lab), n in zip(pairs.T.tolist(), counts.tolist(), strict=True):
        if c not in best or n > best[c][0]:
            best[c] = (n, lab)
    return {c: v[1] for c, v in best.items()}


def audit(
    labels: dict[str, str | None], cur: dict[str, str | None], cand: dict[str, str | None]
) -> dict[str, Any]:
    ids = [i for i in labels if cur.get(i) is not None and cand.get(i) is not None]
    skipped = len(labels) - len(ids)
    lab = np.unique([labels[i] for i in ids], return_inverse=True)[1]
    c0 = np.unique([cur[i] for i in ids], return_inverse=True)[1]
    c1 = np.unique([cand[i] for i in ids], return_inverse=True)[1]
    pair, cnt = np.unique(np.stack([c0, c1]), axis=1, return_counts=True)
    succ: dict[int, tuple[int, int]] = {}
    for (a, b), n in zip(pair.T.tolist(), cnt.tolist(), strict=True):
        if a not in succ or n > succ[a][0]:
            succ[a] = (n, b)
    moved = np.array([succ[a][1] != b for a, b in zip(c0.tolist(), c1.tolist(), strict=True)])
    maj0, maj1 = _majority(c0, lab), _majority(c1, lab)
    old_ok = np.array([maj0[a] == t for a, t in zip(c0.tolist(), lab.tolist(), strict=True)])
    new_ok = np.array([maj1[b] == t for b, t in zip(c1.tolist(), lab.tolist(), strict=True)])
    outcome = Counter(
        {
            'fixed': int((moved & ~old_ok & new_ok).sum()),
            'broken': int((moved & old_ok & ~new_ok).sum()),
            'both_right': int((moved & old_ok & new_ok).sum()),
            'both_wrong': int((moved & ~old_ok & ~new_ok).sum()),
        }
    )
    return {
        'compared': len(ids),
        'skipped_unlabelled_or_unassigned': skipped,
        'moved': int(moved.sum()),
        'moved_fraction': float(moved.mean()) if ids else None,
        **outcome,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--current', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    args = ap.parse_args()
    labels = _read(args.labels, 'label')
    print(
        json.dumps(
            audit(labels, _read(args.current, 'cluster_id'), _read(args.candidate, 'cluster_id')),
            indent=2,
        )
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
