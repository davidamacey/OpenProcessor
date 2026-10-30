#!/usr/bin/env python3
"""Does a change to the ``gpu_image_prep`` model alter what it answers?

``capture`` streams each JPEG straight to the model (so only the model is measured, not the API)
and keeps every output tensor; ``compare`` sets two captures side by side. Run it against the
model before and after a change, on the same images and the same Triton engines:

    python scripts/bench/gpu_prep_parity.py capture --url 127.0.0.1:4601 --images list.txt \\
        --concurrency 1 --out before.npz
    python scripts/bench/gpu_prep_parity.py capture ... --out after.npz
    python scripts/bench/gpu_prep_parity.py compare before.npz after.npz

At concurrency 1 a change that does not touch arithmetic must give identical bytes for
everything but the PE vectors, which may differ by the model's own run-to-run floor (compare two
captures of the same model to measure it). At higher concurrency detections are paired by score
and the table reports the spread against that floor.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import numpy as np


OUTPUTS = (
    'status',
    'size',
    'letterbox',
    'det_boxes',
    'det_scores',
    'det_classes',
    'whole_embedding',
    'crop_embeddings',
    'blur_sums',
    'crop_jpegs',
)
EXACT = ('status', 'size', 'letterbox', 'det_classes', 'blur_sums')


def read_list(path: Path) -> list[Path]:
    rows = [ln.strip() for ln in path.read_text().splitlines()]
    return [Path(r) for r in rows if r and not r.startswith('#')]


async def _one(client: Any, data: bytes, sem: asyncio.Semaphore) -> dict[str, np.ndarray]:
    from tritonclient.grpc import InferInput

    inp = InferInput('jpeg', [len(data)], 'UINT8')
    inp.set_data_from_numpy(np.frombuffer(data, np.uint8))

    async def requests() -> Any:
        yield {'model_name': 'gpu_image_prep', 'inputs': [inp]}

    async with sem:
        async for result, error in client.stream_infer(requests(), stream_timeout=120.0):
            if error is not None:
                raise error
            return {n: result.as_numpy(n) for n in OUTPUTS if result.as_numpy(n) is not None}
    raise RuntimeError('stream closed without a response')


async def capture(url: str, images: list[Path], concurrency: int) -> list[dict[str, np.ndarray]]:
    from tritonclient.grpc.aio import InferenceServerClient

    client = InferenceServerClient(url=url)
    sem = asyncio.Semaphore(concurrency)
    try:
        return list(await asyncio.gather(*(_one(client, p.read_bytes(), sem) for p in images)))
    finally:
        await client.close()


def save(path: Path, images: list[Path], rows: list[dict[str, np.ndarray]]) -> None:
    arrays: dict[str, np.ndarray] = {'names': np.array([p.name for p in images])}
    for i, row in enumerate(rows):
        for name, arr in row.items():
            arrays[f'{i}/{name}'] = np.array(arr, dtype=object) if arr.dtype == object else arr
    np.savez_compressed(path, allow_pickle=True, **arrays)


def load(path: Path) -> list[dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=True)
    n = len(z['names'])
    return [{k.split('/', 1)[1]: z[k] for k in z.files if k.startswith(f'{i}/')} for i in range(n)]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 if d == 0 else float(a @ b / d)


def _status(row: dict[str, np.ndarray]) -> str:
    v = row['status'][0]
    return v.decode() if isinstance(v, bytes) else str(v)


def _by_score(row: dict[str, np.ndarray]) -> np.ndarray:
    return (
        np.argsort(-row['det_scores'], kind='stable') if 'det_scores' in row else np.array([], int)
    )


def compare_pair(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float | bool]:
    """Differences of one image's two answers."""
    out: dict[str, float | bool] = {'status_equal': bool(np.array_equal(a['status'], b['status']))}
    if not out['status_equal'] or _status(a) != 'ok':
        return out
    oa, ob = _by_score(a), _by_score(b)
    out['exact_equal'] = all(
        np.array_equal(
            a[k][oa] if k == 'det_classes' else a[k],
            b[k][ob] if k == 'det_classes' else b[k],
        )
        for k in EXACT
        if k != 'blur_sums'
    )
    out['same_detections'] = a['det_boxes'].shape == b['det_boxes'].shape
    if out['same_detections']:
        out['box_max_abs'] = (
            float(np.abs(a['det_boxes'][oa] - b['det_boxes'][ob]).max()) if len(oa) else 0.0
        )
        out['blur_equal'] = bool(
            np.array_equal(a['blur_sums'][np.r_[0, oa + 1]], b['blur_sums'][np.r_[0, ob + 1]])
        )
        ja, jb = a['crop_jpegs'][oa], b['crop_jpegs'][ob]
        out['crop_jpeg_equal'] = all(bytes(x) == bytes(y) for x, y in zip(ja, jb, strict=True))
        out['whole_cos'] = _cos(a['whole_embedding'], b['whole_embedding'])
        ca, cb = a['crop_embeddings'][oa], b['crop_embeddings'][ob]
        out['crop_cos_min'] = min((_cos(x, y) for x, y in zip(ca, cb, strict=True)), default=1.0)
        out['bytes_equal'] = bool(
            out['exact_equal']
            and out['box_max_abs'] == 0.0
            and out['blur_equal']
            and out['crop_jpeg_equal']
        )
    return out


def summarize(rows: list[dict[str, float | bool]]) -> dict[str, float]:
    ok = [r for r in rows if 'same_detections' in r]
    same = [r for r in ok if r['same_detections']]
    return {
        'images': len(rows),
        'status_differs': sum(not r['status_equal'] for r in rows),
        'detections_differ': len(ok) - len(same),
        'bytes_equal': sum(bool(r.get('bytes_equal')) for r in same),
        'blur_differs': sum(not r['blur_equal'] for r in same),
        'jpeg_differs': sum(not r['crop_jpeg_equal'] for r in same),
        'box_max_abs': max((float(r['box_max_abs']) for r in same), default=0.0),
        'whole_cos_min': min((float(r['whole_cos']) for r in same), default=1.0),
        'crop_cos_min': min((float(r['crop_cos_min']) for r in same), default=1.0),
        'compared': len(same),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('--url', required=True, help='Triton gRPC host:port')
    cap.add_argument('--images', type=Path, required=True, help='file with one image path per line')
    cap.add_argument('--limit', type=int, default=0)
    cap.add_argument('--concurrency', type=int, default=1)
    cap.add_argument('--out', type=Path, required=True)
    cmp_ = sub.add_parser('compare')
    cmp_.add_argument('a', type=Path)
    cmp_.add_argument('b', type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'capture':
        images = read_list(args.images)[: args.limit or None]
        save(args.out, images, asyncio.run(capture(args.url, images, args.concurrency)))
        print(f'{len(images)} images captured to {args.out}')
        return 0
    ra, rb = load(args.a), load(args.b)
    if len(ra) != len(rb):
        print('captures cover different image counts', file=sys.stderr)
        return 2
    for key, value in summarize([compare_pair(x, y) for x, y in zip(ra, rb, strict=True)]).items():
        print(f'{key:20s} {value}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
