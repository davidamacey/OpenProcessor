"""Turn one or more ``*_results.json`` files from bench_driver.py into a markdown table.

tabulate_results.py out/20k_results.json out/100k_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


HEAD = (
    '| Method | Precision | Recall | False merge | False split | Split recovery | Unassigned '
    '| Clusters | Stability ARI | Wall s | Peak RSS MB | GPU MB |'
)


def fmt(v: object, nd: int = 3) -> str:
    if v is None:
        return 'n/a'
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def main() -> int:
    for path in map(Path, sys.argv[1:]):
        print(f'\n{path.stem}\n\n{HEAD}\n|' + '---|' * 12)
        for r in json.loads(path.read_text()):
            if 'error' in r:
                print(f'| {r["name"]} | {r["error"]} |' + ' |' * 10)
                continue
            cells = [
                fmt(r[c])
                for c in (
                    'pair_precision',
                    'pair_recall',
                    'false_merge_rate',
                    'false_split_rate',
                    'correct_split_recovery',
                    'unassigned_fraction',
                )
            ]
            print(
                f'| {r["name"]} | {" | ".join(cells)} | {fmt(r["clusters"])} '
                f'| {fmt(r["stability_ari"])} | {fmt(r["wall_s"], 1)} '
                f'| {fmt(r["peak_rss_mb"], 0)} | {fmt(r.get("gpu_peak_mb"), 0)} |'
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
