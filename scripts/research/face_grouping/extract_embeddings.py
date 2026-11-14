"""Send every image of a labelled face set through ``POST /faces/recognize`` and store the results.

    python -I extract_embeddings.py <set_dir> <out.npz> [--url http://127.0.0.1:4603] [--clients 8]

``<set_dir>`` holds ``labels.jsonl`` (one ``{"identity", "path"}`` row per image, path relative to the
directory). The identity is the row's ``identity`` field: the directory name for LFW and the file-name
prefix before the trailing index for CALFW and CPLFW (as written by ``prepare_labelled_benchmark.py``).
Every detected face is kept (box and landmarks are normalised 0-1, as the route returns them);
choosing the face the set is about is done later by ``face_table.main_face``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np


def recognize(client: httpx.Client, url: str, path: Path) -> dict:
    with path.open('rb') as fh:
        resp = client.post(f'{url}/faces/recognize', files={'image': (path.name, fh, 'image/jpeg')})
    resp.raise_for_status()
    body: dict = resp.json()
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('set_dir', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--url', default='http://127.0.0.1:4603')
    ap.add_argument('--clients', type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(line) for line in (args.set_dir / 'labels.jsonl').read_text().splitlines()]
    paths = [args.set_dir / r['path'] for r in rows]
    with httpx.Client(timeout=120) as client, ThreadPoolExecutor(args.clients) as pool:
        results = list(pool.map(lambda p: recognize(client, args.url, p), paths))

    face_img, box, lm, score, quality, emb = [], [], [], [], [], []
    width = np.zeros(len(rows), dtype=np.int32)
    height = np.zeros(len(rows), dtype=np.int32)
    for i, res in enumerate(results):
        width[i], height[i] = res['image']['width'], res['image']['height']
        for f, e in zip(res['faces'], res['embeddings'], strict=True):
            face_img.append(i)
            box.append(f['box'])
            lm.append(f['landmarks'])
            score.append(f['score'])
            quality.append(f.get('quality') if f.get('quality') is not None else np.nan)
            emb.append(e)
    np.savez_compressed(
        args.out,
        files=np.array([r['path'] for r in rows]),
        identity=np.array([r['identity'] for r in rows]),
        width=width,
        height=height,
        face_img=np.array(face_img, dtype=np.int64),
        box=np.array(box, dtype=np.float32),
        landmarks=np.array(lm, dtype=np.float32),
        score=np.array(score, dtype=np.float32),
        quality=np.array(quality, dtype=np.float32),
        emb=np.array(emb, dtype=np.float32),
    )
    print(f'{args.set_dir.name}: {len(rows)} images, {len(face_img)} faces', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
