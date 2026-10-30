"""Class-level P@10 on the labelled crops of the Z2 vehicle set (FG-1's shared-crop-path check).

Neighbours from the same source image are skipped, so a crop is not matched to its photo-mates.
Both variants of a model are scored the same way: only the difference matters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.research.fine_grained.data import run_info


def class_p10(vec: np.ndarray, label: np.ndarray, group: np.ndarray, k: int = 10) -> float:
    hits = total = 0
    for i0 in range(0, len(vec), 1024):
        sim = vec[i0 : i0 + 1024] @ vec.T
        same_photo = group[i0 : i0 + 1024, None] == group[None, :]
        sim[same_photo] = -np.inf
        top = np.argsort(-sim, axis=1, kind='stable')[:, :k]
        hits += int((label[top] == label[i0 : i0 + 1024, None]).sum())
        total += top.size
    return hits / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--emb', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--keys', nargs='+', required=True, help='MODEL__VARIANT')
    args = ap.parse_args()
    emb = Path(args.emb)
    meta = [json.loads(x) for x in (emb / 'meta_z2.jsonl').read_text().splitlines() if x]
    label, group = np.array([m['label'] for m in meta]), np.array([m['group'] for m in meta])
    out = {**run_info(), 'crops': len(meta), 'class_p10': {}}
    for key in args.keys:
        out['class_p10'][key] = class_p10(np.load(emb / f'{key}__z2.npy'), label, group)
        print(key, out['class_p10'][key])
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
