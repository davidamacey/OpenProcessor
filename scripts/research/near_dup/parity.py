"""nd:P0 parity of the research encoders.

The registered gate compares venv vectors with product (TensorRT) vectors of the same images, which needs a
running product stack. Two modes:

* ``precision``: the venv's own fp16-autocast encoding against its fp32 encoding of the same images
  (a substitute that bounds the precision part of the gap; it does not test the product engine);
* ``compare``: two ``encode.py`` outputs of the same list (for example venv versus
  ``scripts/research/fetch_project_vectors.py`` product vectors), median and minimum row cosine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.research.near_dup import encode


def row_cosine(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {
        'n': float(len(cos)),
        'median': float(np.median(cos)),
        'mean': float(np.mean(cos)),
        'min': float(np.min(cos)),
        'p1': float(np.percentile(cos, 1)),
    }


def precision(
    paths: list[str], space: str, device: str, batch: int, threads: int
) -> dict[str, float]:
    import torch

    from scripts.research.fine_grained import models

    enc = models.load(encode.SPACES[space], device)
    half, full = [], []
    for x in encode.batches(paths, space, batch, threads):
        half.append(enc.encode(x))
        with torch.inference_mode():
            t = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255
            out = enc._forward((t - enc._mean) / enc._std).float()
            full.append(torch.nn.functional.normalize(out, dim=-1).cpu().numpy())
    return row_cosine(np.concatenate(half), np.concatenate(full))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('precision')
    p.add_argument('--paths', type=Path, required=True)
    p.add_argument('--space', choices=sorted(encode.SPACES), required=True)
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('compare')
    c.add_argument('--a', type=Path, required=True)
    c.add_argument('--b', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'precision':
        paths = [ln for ln in args.paths.read_text().splitlines() if ln][: args.limit]
        result = {
            'space': args.space,
            'fp16_vs_fp32': precision(paths, args.space, args.device, 32, 8),
        }
    else:
        result = row_cosine(np.load(args.a), np.load(args.b))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    sys.stdout.write(json.dumps(result) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
