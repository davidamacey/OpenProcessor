#!/usr/bin/env python3
"""Goldens for the native frame engine's ``ocr`` branch (plan 17 section 5 N1, row N1c).

Runs the Python engine's OCR functions (``frame_engine/ocr/``) and the OpenCV and pyclipper calls
they make on inputs this script fixes, and writes what they answered under
``native/frame_engine/tests/golden/``:

* ``ocr_db.json``: probability maps described by integer shapes (rotated bars, ellipses, rings
  with holes, one-pixel lines and dots, low noise; ``raster`` below draws them, and the Rust test
  draws them with the same integer arithmetic), each with OpenCV's contours, every contour's
  ``_mini_box`` and ``_score``, and ``boxes_from_map``'s boxes and scores; plus float rectangles
  through ``_unclip`` then ``_mini_box`` (pyclipper's union included);
* ``ocr_lines.json``: ``det_size`` over a grid of sizes and limits, ``crop_dims``, ``rec_ratio``,
  ``rec_map`` and ``taps_for`` of integer quadrilaterals (small, tall, turned, up to 8000
  pixels), ``plan_batches`` and ``sort_reading_order`` of random inputs;
* ``ocr_ctc.json``: ``CtcDecoder`` over the recogniser's real dictionary
  (``models/paddleocr_rec_trt/ppocrv6_dict.txt``): rows with blanks, repeats, indices past the
  dictionary and empty rows, the text and both scores.

Floats are written as their IEEE bit patterns, so equal goldens are equal bits. The data is
committed; the Rust tests read it. This script is deleted with the Python engine at the switch
(plan 17 N3). ``--check`` regenerates into memory and compares with the files.
"""

from __future__ import annotations

import json
import random
import struct
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'native' / 'frame_engine' / 'tests' / 'golden'
DICTIONARY = ROOT / 'models' / 'paddleocr_rec_trt' / 'ppocrv6_dict.txt'


def _setup() -> None:
    """Make the repository importable; ``frame_engine`` imports CuPy only for its kernels."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.modules.setdefault('cupy', types.ModuleType('cupy'))


def f32(x: float) -> int:
    return int(struct.unpack('<I', struct.pack('<f', x))[0])


def f64(x: float) -> int:
    return int(struct.unpack('<Q', struct.pack('<d', x))[0])


# ---------------------------------------------------------------------------------- maps


def raster(case: dict[str, Any]) -> list[int]:
    """The map's levels (0..255, row after row): integer arithmetic only, so the Rust test draws
    the same map. A pixel centre is ``(2x + 1, 2y + 1)`` in doubled coordinates."""
    w, h = case['w'], case['h']
    out = []
    for y in range(h):
        for x in range(w):
            px, py = 2 * x + 1, 2 * y + 1
            q = 0
            for s in case['shapes']:
                rx, ry = px - s['cx'], py - s['cy']
                a, b = s['a'], s['b']
                if s['kind'] == 'bar':
                    n2 = s['dx'] * s['dx'] + s['dy'] * s['dy']
                    u = rx * s['dx'] + ry * s['dy']
                    v = -rx * s['dy'] + ry * s['dx']
                    inside = u * u <= a * a * n2 and v * v <= b * b * n2
                else:
                    inside = rx * rx * b * b + ry * ry * a * a <= a * a * b * b
                    if s['kind'] == 'ring':
                        ia, ib = a - s['inner'], b - s['inner']
                        inside = (
                            inside and rx * rx * ib * ib + ry * ry * ia * ia > ia * ia * ib * ib
                        )
                if inside:
                    q = max(q, s['level'] - ((x + 2 * y) % 5) * 4)
            if case['noise']:
                hsh = ((x * 73856093) ^ (y * 19349663) ^ (case['seed'] * 83492791)) & 0xFFFF
                q = min(255, q + hsh % case['noise'])
            out.append(q)
    return out


def _shape(rng: random.Random, w: int, h: int) -> dict[str, Any]:
    kind = rng.choice(('bar', 'bar', 'bar', 'ellipse', 'ring', 'line', 'dot'))
    s: dict[str, Any] = {
        'cx': rng.randint(-4, 2 * w + 4),
        'cy': rng.randint(-4, 2 * h + 4),
        'level': rng.randint(60, 255),
    }
    if kind in ('bar', 'line'):
        s.update(kind='bar', dx=rng.randint(-8, 8), dy=rng.choice((-2, -1, 0, 0, 0, 1, 2)))
        if s['dx'] == 0 and s['dy'] == 0:
            s['dx'] = 1
        s['a'] = rng.randint(4, w)
        s['b'] = 1 if kind == 'line' else rng.randint(2, max(3, h // 4))
    elif kind == 'dot':
        s.update(kind='ellipse', a=rng.randint(1, 3), b=rng.randint(1, 3))
    else:
        s.update(kind=kind, a=rng.randint(6, max(7, w // 2)), b=rng.randint(6, max(7, h // 2)))
        if kind == 'ring':
            s['inner'] = rng.randint(2, max(2, min(s['a'], s['b']) - 2))
    return s


def _maps() -> list[dict[str, Any]]:
    rng = random.Random(20261115)
    cases = []
    for k in range(160):
        w, h = rng.randint(8, 112), rng.randint(8, 80)
        cases.append(
            {
                'w': w,
                'h': h,
                'seed': k,
                'noise': rng.choice((0, 0, 0, 0, 0, 30, 60)),
                'shapes': [_shape(rng, w, h) for _ in range(rng.randint(0, 7))],
                'dest': [rng.randint(w // 2 + 1, w * 9), rng.randint(h // 2 + 1, h * 9)],
                'params': rng.choice(
                    (
                        (0.2, 0.45, 1.4),
                        (0.2, 0.45, 1.4),
                        (0.3, 0.6, 2.0),
                        (0.1, 0.2, 1.5),
                        (0.5, 0.5, 1.0),
                    )
                ),
            }
        )
    # degenerate: the whole map on, one pixel, a one-pixel border
    cases.append(
        {
            'w': 12,
            'h': 9,
            'seed': 900,
            'noise': 0,
            'dest': [120, 90],
            'params': (0.2, 0.45, 1.4),
            'shapes': [
                {'kind': 'bar', 'cx': 12, 'cy': 9, 'a': 40, 'b': 40, 'dx': 1, 'dy': 0, 'level': 230}
            ],
        }
    )
    cases.append(
        {
            'w': 5,
            'h': 5,
            'seed': 901,
            'noise': 0,
            'dest': [5, 5],
            'params': (0.2, 0.45, 1.4),
            'shapes': [{'kind': 'ellipse', 'cx': 5, 'cy': 5, 'a': 1, 'b': 1, 'level': 255}],
        }
    )
    return cases


def _db_case(case: dict[str, Any]) -> dict[str, Any]:
    import cv2
    import numpy as np

    from frame_engine.ocr import db_boxes

    w, h = case['w'], case['h']
    pred = (np.array(raster(case), np.float32) / np.float32(255)).reshape(h, w)
    thresh, box_thresh, ratio = case['params']
    mask = (pred > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    per = []
    for c in contours:
        pts, side = db_boxes._mini_box(c)
        arr = np.array(pts)
        per.append(
            {
                'points': c.reshape(-1, 2).tolist(),
                'mini': [f32(float(v)) for v in arr.reshape(-1)],
                'side': f32(side),
                'score': f64(db_boxes._score(pred, arr)),
            }
        )
    boxes, scores = db_boxes.boxes_from_map(
        pred, case['dest'][0], case['dest'][1], db_boxes.DbParams(thresh, box_thresh, ratio)
    )
    return {
        **case,
        'params': list(case['params']),
        'contours': per,
        'boxes': boxes.reshape(-1, 8).tolist(),
        'scores': [f32(float(s)) for s in scores],
    }


def _unclip_cases() -> list[dict[str, Any]]:
    import cv2
    import numpy as np

    from frame_engine.ocr import db_boxes

    rng = random.Random(7)
    out = []
    for _ in range(1500):
        size = rng.choice((3.0, 4.0, 8.0, 40.0, 300.0))
        rect = (
            (rng.uniform(0, 900), rng.uniform(0, 900)),
            (rng.uniform(3.0, size * 3), rng.uniform(3.0, size)),
            rng.uniform(-90.0, 0.0),
        )
        pts = np.array(cv2.boxPoints(rect), np.float32)
        ratio = rng.choice((1.4, 1.4, 1.5, 2.0, 0.5))
        grown = db_boxes._unclip(pts, ratio)
        row: dict[str, Any] = {'corners': [f32(float(v)) for v in pts.reshape(-1)], 'ratio': ratio}
        if grown is not None:
            quad, side = db_boxes._mini_box(grown.reshape(-1, 1, 2))
            row['mini'] = [f32(float(v)) for v in np.array(quad).reshape(-1)]
            row['side'] = f32(side)
        out.append(row)
    return out


# ---------------------------------------------------------------------------------- lines


def _quads(rng: random.Random) -> list[list[list[int]]]:
    import cv2
    import numpy as np

    out = []
    for _ in range(600):
        scale = rng.choice((2, 10, 60, 400, 3000, 8000))
        w = rng.uniform(1, scale)
        h = rng.uniform(1, max(2.0, scale / rng.choice((1, 3, 10))))
        if rng.random() < 0.3:
            w, h = h, w  # tall lines get the quarter turn
        rect = ((rng.uniform(0, 9000), rng.uniform(0, 9000)), (w, h), rng.uniform(-20.0, 20.0))
        out.append(np.round(cv2.boxPoints(rect)).astype(int).tolist())
    out.append([[5, 5], [5, 5], [5, 5], [5, 5]])
    out.append([[0, 0], [3, 0], [3, 1], [0, 1]])
    return out


def _lines() -> dict[str, Any]:
    import numpy as np

    from frame_engine.ocr import batching, crop_geometry, db_boxes, det_input

    rng = random.Random(11)
    det = []
    for h, w in [
        (1, 1),
        (31, 33),
        (480, 640),
        (640, 480),
        (3000, 4000),
        (6000, 8000),
        (17, 9000),
        (9000, 17),
        (736, 736),
        (1080, 1920),
        (100, 3000),
    ]:
        for side, limit, max_side in [
            (736, 'min', 4000),
            (960, 'max', 4000),
            (64, 'max', 1920),
            (1920, 'min', 1920),
            (32, 'min', 32),
            (960, 'min', 1000),
        ]:
            det.append(
                [h, w, side, limit, max_side, *det_input.det_size(h, w, side, limit, max_side)]
            )
    quads = []
    for q in _quads(rng):
        arr = np.array(q)
        row: dict[str, Any] = {'quad': q, 'dims': list(crop_geometry.crop_dims(arr))}
        ratio = crop_geometry.rec_ratio(arr)
        row['ratio'] = None if ratio is None else f64(ratio)
        if ratio is not None:
            for width in sorted({batching.resized_width(ratio, 2048), 48, 320}):
                g = crop_geometry.rec_map(arr, width)
                row.setdefault('maps', []).append(
                    {
                        'width': width,
                        'map': [f32(float(v)) for v in g.astype(np.float32).reshape(9)],
                        'taps': [
                            list(crop_geometry.taps_for(g, width, cap)) for cap in (1, 2, 4, 8)
                        ],
                    }
                )
        quads.append(row)
    batches = []
    for _ in range(60):
        ratios = [
            rng.choice((rng.uniform(0.2, 3), rng.uniform(3, 60), 2.0))
            for _ in range(rng.randint(0, 40))
        ]
        batch = rng.randint(1, 16)
        planned = batching.plan_batches(ratios, batch, max_width=2048)
        batches.append(
            {
                'ratios': [f64(r) for r in ratios],
                'batch': batch,
                'out': [[b.indices, b.width] for b in planned],
            }
        )
    orders = []
    for _ in range(80):
        n = rng.randint(0, 25)
        boxes = []
        for _ in range(n):
            x, y = rng.randint(0, 400), rng.choice((rng.randint(0, 400), rng.randint(100, 115)))
            boxes.append([[x, y], [x + 9, y], [x + 9, y + 9], [x, y + 9]])
        order = db_boxes.sort_reading_order(np.array(boxes, np.int32).reshape(-1, 4, 2))
        orders.append({'boxes': boxes, 'order': order.tolist()})
    return {'det_size': det, 'quads': quads, 'batches': batches, 'reading_order': orders}


# ---------------------------------------------------------------------------------- ctc


def _ctc() -> dict[str, Any]:
    import numpy as np

    from frame_engine.ocr.ctc import CtcDecoder

    decoder = CtcDecoder.from_file(DICTIONARY)
    n = len(decoder.characters)
    rng = np.random.default_rng(5)
    rows = []
    for k in range(400):
        t = int(rng.integers(0, 60))
        pick = rng.random(t)
        idx = np.where(pick < 0.4, 0, rng.integers(1, n + 40, t)).astype(np.int32)
        if k % 3 == 0 and t > 2:
            idx[1:] = np.where(rng.random(t - 1) < 0.5, idx[:-1], idx[1:])  # runs of repeats
        scores = rng.random(t).astype(np.float32)
        text, mean, low = decoder.decode(idx, scores)
        rows.append(
            {
                'indices': idx.tolist(),
                'scores': [f32(float(s)) for s in scores],
                'text': text,
                'mean': f32(mean),
                'min': f32(low),
            }
        )
    return {'classes': n, 'rows': rows}


# ---------------------------------------------------------------------------------- files


def _dump(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n'


def render() -> dict[str, str]:
    _setup()
    sources = ['frame_engine/ocr/', 'OpenCV 5.0.0', 'pyclipper 1.4.0']
    head = {'generated_by': 'scripts/codegen/frame_engine_text_goldens.py', 'sources': sources}
    return {
        'ocr_db.json': _dump(
            {**head, 'maps': [_db_case(c) for c in _maps()], 'unclip': _unclip_cases()}
        ),
        'ocr_lines.json': _dump({**head, **_lines()}),
        'ocr_ctc.json': _dump({**head, **_ctc()}),
    }


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import frame_engine_goldens

    return int(
        frame_engine_goldens.main(argv, render, 'scripts/codegen/frame_engine_text_goldens.py')
    )


if __name__ == '__main__':
    raise SystemExit(main())
