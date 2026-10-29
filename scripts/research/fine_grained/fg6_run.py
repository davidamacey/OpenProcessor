"""FG-6 runner: ``python -m scripts.research.fine_grained.fg6_run --emb DIR --model NAME --out FILE``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.research.fine_grained import fg6
from scripts.research.fine_grained.data import load_universe, load_vectors, run_info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--emb', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--variant', default='centre')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    u = load_universe(Path(args.emb))
    vec = load_vectors(Path(args.emb), args.model, args.variant)
    out = {**run_info(), 'model': args.model, 'variant': args.variant}
    for f in (1, 2):
        out[f'fold{f}'] = fg6.run_fold(vec, u, f)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    print(json.dumps({k: out[k]['auroc_by_make'] for k in ('fold1', 'fold2')}, indent=1))


if __name__ == '__main__':
    main()
