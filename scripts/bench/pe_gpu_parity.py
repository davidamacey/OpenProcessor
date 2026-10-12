"""Do the cards agree? PE embeddings of the same crops, captured on one GPU at a time.

    pe_gpu_parity.py capture --triton-url triton-server:8001 --images DIR --label gpu1 \\
        --out artifacts_local/bench/pe_gpu/gpu1.npz [--count 256]
    pe_gpu_parity.py compare A.npz B.npz [C.npz ...]

``capture`` encodes the first ``--count`` pinned images (resize, centre crop, uint8, exactly the
ingest preprocessing) through ``pe_image_encoder`` in batches of 32 and stores the vectors. Run
it once per card against a Triton that sees only that card (``TRITON_GPU_ID=<n>``), from a
container on the Triton network that has ``tritonclient`` (the API image). ``compare`` prints,
for every pair, the cosine between the two vectors of the same crop and how many are bit-identical.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BATCH = 32
MODEL = 'pe_image_encoder'


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Cosine and absolute difference between row i of ``a`` and row i of ``b``."""
    if a.shape != b.shape:
        raise ValueError(f'shape {a.shape} against {b.shape}: not the same crops')
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    cosine = (a64 * b64).sum(axis=1) / (np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1))
    diff = np.abs(a64 - b64)
    return {
        'n': float(len(a)),
        'cosine_min': float(cosine.min()),
        'cosine_mean': float(cosine.mean()),
        'max_abs_diff': float(diff.max()),
        'identical': float((diff.max(axis=1) == 0).sum()),
    }


def capture(args: argparse.Namespace) -> None:
    from tritonclient import grpc  # only inside the API image

    from src.services.detection.pe_preprocess import whole_frame_chw

    files = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )
    crops = [c for c in (whole_frame_chw(str(p)) for p in files[: args.count]) if c is not None]
    client = grpc.InferenceServerClient(args.triton_url)
    vectors = []
    for start in range(0, len(crops), BATCH):
        batch = np.stack(crops[start : start + BATCH])
        inp = grpc.InferInput('images', list(batch.shape), 'UINT8')
        inp.set_data_from_numpy(batch)
        out = client.infer(MODEL, [inp]).as_numpy('image_embeddings')
        vectors.append(out.astype(np.float32))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, vectors=np.concatenate(vectors), label=args.label)
    print(f'{args.label}: {len(crops)} crops -> {args.out}')


def compare(args: argparse.Namespace) -> None:
    loaded = {p.stem: np.load(p)['vectors'] for p in args.files}
    rows = {
        f'{x} vs {y}': pair_stats(loaded[x], loaded[y])
        for x, y in itertools.combinations(loaded, 2)
    }
    print(json.dumps(rows, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--triton-url', required=True, help='gRPC host:port')
    c.add_argument('--images', type=Path, required=True)
    c.add_argument('--label', required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--count', type=int, default=256)
    c.set_defaults(fn=capture)
    m = sub.add_parser('compare')
    m.add_argument('files', type=Path, nargs='+')
    m.set_defaults(fn=compare)
    args = p.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
