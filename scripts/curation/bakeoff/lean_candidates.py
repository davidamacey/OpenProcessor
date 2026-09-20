#!/usr/bin/env python3
"""Sample motorcycle plate crops and tile them for visual lean-angle review.

Phase-3 lean-angle prototype helper. Pulls a random sample of confirmed-plate
crops on a leaning vehicle class (default ``sportbike`` -- the hardest-leaning),
crops a padded region around each plate from the *source* frame, labels it with
an index, and tiles them into montage image(s) for visual angle assessment. Also
writes per-candidate higher-res crops and a manifest mapping index -> crop
metadata, so a computed angle can be matched back to each plate.

    .venv/bin/python scripts/curation/bakeoff/lean_candidates.py --n 36 --class sportbike

Outputs under ``/tmp/lean/``: ``montage_*.png``, ``crops/NN.png``, ``manifest.json``.
Reads images directly from the NAS paths in ``image_path`` (readable on host).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import requests

from src.config import get_curation_config, get_region_fields
from src.config.region_state import RegionStatus


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('lean_candidates')

OS_URL = 'http://localhost:4607'
OUT = Path('/tmp/lean')
CROPS_INDEX = get_curation_config().items_index


def sample_crops(n: int, vclass: str, seed: int) -> list[dict]:
    """Random sample of confirmed-plate crops on the given vehicle class."""
    F = get_region_fields()
    body = {
        'size': n,
        '_source': ['crop_id', 'image_path', 'hdd_source', F.bbox_norm, F.text],
        'query': {
            'function_score': {
                'query': {
                    'bool': {
                        'must': [
                            {'term': {'class_name.keyword': vclass}},
                            {'term': {f'{F.status}.keyword': RegionStatus.DETECTED}},
                            {'exists': {'field': F.bbox_norm}},
                        ]
                    }
                },
                'random_score': {'seed': seed, 'field': '_seq_no'},
            }
        },
    }
    resp = requests.post(f'{OS_URL}/{CROPS_INDEX}/_search', json=body, timeout=90).json()
    return [{**h['_source'], '_id': h['_id']} for h in resp.get('hits', {}).get('hits', [])]


def crop_plate(
    image_path: str, bbox: list[float], *, pad: float, longest: int
) -> np.ndarray | None:
    """Crop a padded region around the source-frame plate bbox; longest side scaled."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0.0, x1 - bw * pad), min(1.0, x2 + bw * pad)
    y1, y2 = max(0.0, y1 - bh * pad), min(1.0, y2 + bh * pad)
    cx1, cy1, cx2, cy2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    scale = longest / max(ch, cw)
    return cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))


def to_tile(crop: np.ndarray, idx: int, size: int) -> np.ndarray:
    """Center a crop on a square canvas and stamp its index."""
    canvas = np.full((size, size, 3), 38, np.uint8)
    s = size / max(crop.shape[:2])
    rc = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
    yh, xw = rc.shape[:2]
    yo, xo = (size - yh) // 2, (size - xw) // 2
    canvas[yo : yo + yh, xo : xo + xw] = rc
    cv2.rectangle(canvas, (0, 0), (38, 20), (0, 0, 0), -1)
    cv2.putText(
        canvas, str(idx), (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA
    )
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=36)
    ap.add_argument('--class', dest='vclass', default='sportbike')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--pad', type=float, default=0.5)
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--out', type=Path, default=OUT, help='Output dir for montages/crops/manifest.')
    args = ap.parse_args()

    out_dir = args.out
    (out_dir / 'crops').mkdir(parents=True, exist_ok=True)
    rows = sample_crops(args.n, args.vclass, args.seed)
    logger.info('sampled %d %s crops', len(rows), args.vclass)

    manifest: list[dict] = []
    tiles: list[np.ndarray] = []
    for i, r in enumerate(rows):
        bbox = r.get(get_region_fields().bbox_norm)
        if not bbox or len(bbox) != 4:
            continue
        crop = crop_plate(r['image_path'], bbox, pad=args.pad, longest=400)
        if crop is None:
            logger.warning('skip %d: unreadable %s', i, r.get('image_path'))
            continue
        cv2.imwrite(str(out_dir / 'crops' / f'{i:02d}.png'), crop)
        tiles.append(to_tile(crop, i, args.tile))
        manifest.append(
            {
                'idx': i,
                'id': r['_id'],
                'crop_id': r.get('crop_id'),
                'image_path': r['image_path'],
                get_region_fields().bbox_norm: bbox,
                get_region_fields().text: r.get(get_region_fields().text),
            }
        )

    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    cols = args.cols
    per_page = cols * 6
    for p in range(0, len(tiles), per_page):
        chunk = tiles[p : p + per_page]
        rows_n = (len(chunk) + cols - 1) // cols
        grid = np.full((rows_n * args.tile, cols * args.tile, 3), 20, np.uint8)
        for j, t in enumerate(chunk):
            rr, cc = divmod(j, cols)
            grid[rr * args.tile : (rr + 1) * args.tile, cc * args.tile : (cc + 1) * args.tile] = t
        out = out_dir / f'montage_{p // per_page}.png'
        cv2.imwrite(str(out), grid)
        logger.info('wrote %s (%d tiles)', out, len(chunk))
    logger.info('manifest: %s  (%d candidates)', out_dir / 'manifest.json', len(manifest))


if __name__ == '__main__':
    main()
