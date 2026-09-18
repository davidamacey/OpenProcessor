#!/usr/bin/env python3
"""Prototype + figure tool for plate in-plane lean recovery from OCR geometry.

The reusable core lives in ``src.services.detection.region_lean``; this script is the
I/O + PaddleOCR + visualization harness around it. A plate's detector box is
axis-aligned, so it carries no rotation -- the rotation lives in the *text*. We
run PaddleOCR text detection on a context-padded crop, hand the rotated text-line
quads to :func:`plate_lean.estimate_lean` (median longest-edge tilt + a
frontal-vs-oblique spread), and rotate the crop by that angle to level it.

Two modes:

* default -- write ``<review-dir>/ocr_bounds_montage.png`` (detected text quads),
  ``deskew_montage.png`` (each crop rotated flat), and ``lean_results.json``::

      .venv/bin/python scripts/curation/bakeoff/deskew_prototype.py \
          --review-dir docs/paper/lean_review --idx 0,15,2,71

* ``--paper-figure`` -- write a publication panel of before/after pairs with the
  plate text Gaussian-blurred for privacy (``docs/paper/figures/lean_examples.png``).

Reads images from the NAS paths recorded in the ``lean_candidates`` manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.config import get_region_fields
from src.services.detection.region_lean import (
    TextLine,
    deskew,
    estimate_from_lines,
    lines_from_polys,
    text_polys,
)


os.environ.setdefault('DISABLE_MODEL_SOURCE_CHECK', 'True')
os.environ.setdefault('GLOG_minloglevel', '3')
logging.basicConfig(level=logging.INFO, format='%(message)s')
for _n in ('paddle', 'paddlex', 'ppocr', 'PaddleOCR'):
    logging.getLogger(_n).setLevel(logging.ERROR)
logger = logging.getLogger('deskew')


def tight_plate(
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
    crop = img[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]
    if crop.size == 0:
        return None
    s = longest / max(crop.shape[:2])
    return cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))


def detect_lines(det, crop: np.ndarray) -> list[TextLine]:
    """PaddleOCR text detection -> :class:`plate_lean.TextLine` records."""
    return lines_from_polys(text_polys(det, crop))


def draw_quads(img: np.ndarray, lines: list[TextLine]) -> np.ndarray:
    """Draw every detected text quad (green); the longest edge of each is red."""
    out = img.copy()
    for ln in lines:
        pts = ln.quad.astype(int)
        for i in range(len(pts)):
            cv2.line(
                out, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 200, 0), 1, cv2.LINE_AA
            )
        edges = [(ln.quad[i], ln.quad[(i + 1) % len(ln.quad)]) for i in range(len(ln.quad))]
        a, b = max(edges, key=lambda e: float(np.hypot(*(e[1] - e[0]))))
        cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
    return out


def draw_hguides(img: np.ndarray) -> np.ndarray:
    """Faint horizontal reference lines for judging whether a result is level."""
    out = img.copy()
    h, w = out.shape[:2]
    for frac in (0.32, 0.5, 0.68):
        y = int(h * frac)
        cv2.line(out, (0, y), (w, y), (90, 90, 90), 1, cv2.LINE_AA)
    return out


def blur_text(crop: np.ndarray, lines: list[TextLine]) -> np.ndarray:
    """Redact the detected text regions for privacy, keeping plate shape.

    Hard pixelate-then-blur over the text mask so the plate characters (the
    identifying content) are unreadable, while the plate outline and the overall
    tilt of the text band stay visible -- enough to show the lean and its
    correction. Matches the privacy treatment of the near-duplicate figure.
    """
    if not lines:
        return crop.copy()
    h, w = crop.shape[:2]
    small = cv2.resize(crop, (max(1, w // 20), max(1, h // 20)), interpolation=cv2.INTER_LINEAR)
    redacted = cv2.GaussianBlur(
        cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST),
        (0, 0),
        sigmaX=max(3.0, w / 50.0),
    )
    mask = np.zeros((h, w), np.uint8)
    for ln in lines:
        cv2.fillConvexPoly(mask, cv2.convexHull(ln.quad.astype(np.int32)), 255)
    mask = cv2.dilate(mask, np.ones((max(5, h // 30), max(5, h // 30)), np.uint8))
    out = crop.copy()
    out[mask > 0] = redacted[mask > 0]
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 11 + 8, 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _tile(img: np.ndarray, size: int) -> np.ndarray:
    """Center an image on a square canvas for grid layout."""
    canvas = np.full((size, size, 3), 30, np.uint8)
    s = size / max(img.shape[:2])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    yh, xw = r.shape[:2]
    yo, xo = (size - yh) // 2, (size - xw) // 2
    canvas[yo : yo + yh, xo : xo + xw] = r
    return canvas


def _grid(tiles: list[np.ndarray], cols: int, tile: int) -> np.ndarray:
    rows_n = (len(tiles) + cols - 1) // cols
    grid = np.full((rows_n * tile, cols * tile, 3), 20, np.uint8)
    for j, t in enumerate(tiles):
        rr, cc = divmod(j, cols)
        grid[rr * tile : (rr + 1) * tile, cc * tile : (cc + 1) * tile] = t
    return grid


def run_montage(det, manifest: dict, want: list[int], args) -> None:
    """Validation montages + ``lean_results.json`` over the requested indices."""
    befores: list[np.ndarray] = []
    afters: list[np.ndarray] = []
    results: list[dict] = []
    for i in want:
        m = manifest.get(i)
        if m is None:
            continue
        crop = tight_plate(
            m['image_path'], m[get_region_fields().bbox_norm], pad=args.pad, longest=args.tile
        )
        if crop is None:
            logger.warning('idx %d unreadable', i)
            continue
        lines = detect_lines(det, crop)
        est = estimate_from_lines(lines)
        ang = est.angle_deg or 0.0
        flag = '' if est.frontal else '?'
        print(
            f'idx {i:2d}  lean={ang:+6.1f}  spread={est.spread_deg:4.1f}  lines={est.text_lines} {flag}',
            flush=True,
        )
        results.append(
            {
                'idx': i,
                'lean_deg': round(ang, 1) if est.angle_deg is not None else None,
                'spread_deg': round(est.spread_deg, 1),
                'text_lines': est.text_lines,
                'frontal': est.frontal,
                get_region_fields().text: m.get(get_region_fields().text),
                'image_path': m['image_path'],
            }
        )
        tag = f'#{i} {ang:+.0f} s{est.spread_deg:.0f} n{est.text_lines}{flag}'
        befores.append(_tile(label(draw_quads(crop, lines), tag), args.tile))
        afters.append(
            _tile(label(draw_hguides(deskew(crop, ang)), f'#{i} flat {ang:+.0f}{flag}'), args.tile)
        )

    (args.review_dir / 'lean_results.json').write_text(json.dumps(results, indent=2))
    print(f'wrote {args.review_dir / "lean_results.json"} ({len(results)} rows)', flush=True)
    for name, tiles in (('ocr_bounds', befores), ('deskew', afters)):
        if tiles:
            out = args.review_dir / f'{name}_montage.png'
            cv2.imwrite(str(out), _grid(tiles, 6, args.tile))
            print(f'wrote {out} ({len(tiles)} tiles)', flush=True)


def run_paper_figure(det, manifest: dict, want: list[int], args) -> None:
    """Privacy-blurred before/after grid for the paper (``--fig-out``).

    Lays out ``before -> after`` pairs two-per-row so the panel is wide rather
    than tall, which keeps it to one column without crowding the page floats.
    """
    tile, per_row = 300, 2
    pairs: list[np.ndarray] = []
    for i in want:
        m = manifest.get(i)
        if m is None:
            continue
        crop = tight_plate(
            m['image_path'], m[get_region_fields().bbox_norm], pad=args.pad, longest=tile
        )
        if crop is None:
            continue
        lines = detect_lines(det, crop)
        est = estimate_from_lines(lines)
        if not est.frontal:
            continue
        ang = est.angle_deg or 0.0
        safe = blur_text(crop, lines)
        before = _tile(label(safe, f'{ang:+.0f}\xb0'), tile)
        after = _tile(draw_hguides(deskew(safe, ang)), tile)
        arrow = np.full((tile, 22, 3), 20, np.uint8)
        cv2.putText(
            arrow,
            '>',
            (2, tile // 2 + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        pairs.append(np.hstack([before, arrow, after]))
    if not pairs:
        print('no frontal examples to plot', flush=True)
        return
    pw = pairs[0].shape[1]
    hsep = np.full((tile, 18, 3), 20, np.uint8)
    rows: list[np.ndarray] = []
    for r in range(0, len(pairs), per_row):
        group = pairs[r : r + per_row]
        group += [np.full((tile, pw, 3), 20, np.uint8)] * (per_row - len(group))
        strip: list[np.ndarray] = []
        for j, p in enumerate(group):
            if j:
                strip.append(hsep)
            strip.append(p)
        rows.append(np.hstack(strip))
    vsep = np.full((14, rows[0].shape[1], 3), 20, np.uint8)
    stacked: list[np.ndarray] = []
    for j, row in enumerate(rows):
        if j:
            stacked.append(vsep)
        stacked.append(row)
    args.fig_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.fig_out), np.vstack(stacked))
    print(f'wrote {args.fig_out} ({len(pairs)} examples)', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review-dir', type=Path, default=Path('docs/paper/lean_review'))
    ap.add_argument('--idx', default='14,15,20,26,34,3,8,16,22')
    ap.add_argument('--pad', type=float, default=0.30, help='Context pad so OCR det fires.')
    ap.add_argument('--tile', type=int, default=320)
    ap.add_argument(
        '--paper-figure', action='store_true', help='Emit the blurred before/after panel.'
    )
    ap.add_argument('--fig-out', type=Path, default=Path('docs/paper/figures/lean_examples.png'))
    args = ap.parse_args()

    from paddleocr import TextDetection

    det = TextDetection()
    manifest = {m['idx']: m for m in json.loads((args.review_dir / 'manifest.json').read_text())}
    want = [int(x) for x in args.idx.split(',') if x.strip()] or sorted(manifest)
    if args.paper_figure:
        run_paper_figure(det, manifest, want, args)
    else:
        run_montage(det, manifest, want, args)


if __name__ == '__main__':
    main()
