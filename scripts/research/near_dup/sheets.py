"""Visual check sheets for the distinct-original pairs that a cut merges (false-group candidates).

For one space, one split and one cut: every group that holds more than one original gets a sheet with the
originals side by side (up to 6 per sheet, largest groups first). Local scratch only, never committed. The
sheets are read by the coding agent (an agent read, not a blind human or hosted-judge read).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.research.near_dup import group, sweep


TILE = 256


def mixed_groups(labels: np.ndarray, family: np.ndarray) -> list[np.ndarray]:
    """Groups holding more than one family, as arrays of row indices, largest first."""
    out = []
    for g in np.unique(labels):
        rows = np.flatnonzero(labels == g)
        if len(rows) > 1 and len(set(family[rows].tolist())) > 1:
            out.append(rows)
    return sorted(out, key=len, reverse=True)


def sheet(paths: list[str], dest: Path) -> None:
    canvas = Image.new('RGB', (TILE * len(paths), TILE), (255, 255, 255))
    for k, p in enumerate(paths):
        with Image.open(p) as opened:
            tile = opened.convert('RGB')
        tile.thumbnail((TILE, TILE))
        canvas.paste(tile, (k * TILE, 0))
    canvas.save(dest, 'JPEG', quality=85)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--vectors', type=Path, required=True)
    ap.add_argument('--cut', type=float, required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--half', choices=['calibration', 'test'], default='test')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=20)
    args = ap.parse_args(argv)
    rows = sweep.pool_from_manifest(json.loads(args.manifest.read_text()))
    x = np.load(args.vectors.with_suffix('.npy'))
    cal, test = sweep.split_families(1 + max(r['family'] for r in rows), args.seed)
    keep = [i for i, r in enumerate(rows) if r['family'] in (test if args.half == 'test' else cal)]
    pool = sweep.Pool(x[keep], [rows[i] for i in keep])
    labels, _ = group.group_labels(pool.x, pool.edges, args.cut, 'single')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    found = mixed_groups(labels, pool.family)
    for n, members in enumerate(found[: args.limit]):
        per_family: dict[int, int] = {}
        for i in members:
            fam = int(pool.family[i])
            if fam not in per_family or pool.transform[i] == 'orig':
                per_family[fam] = keep[i]
        sheet(
            [rows[i]['path'] for i in list(per_family.values())[:6]],
            args.out_dir / f'mixed_{n:02d}.jpg',
        )
    sys.stdout.write(
        f'{len(found)} mixed groups; {min(len(found), args.limit)} sheets in {args.out_dir}\n'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
