#!/usr/bin/env python3
"""Goldens for the native frame engine's crop side cap (plan 17 section 5 N1a).

The Python engine caps a crop's long side at ``OP_FRAME_ENGINE_CROP_MAX_SIDE`` before its JPEG
encode (``frame_engine/branches/crops.py``, ``CropJpegs._encode``): the size rule below, then
``cv2.resize(..., interpolation=cv2.INTER_AREA)``. This script runs that on fixed crops (a
pattern both languages compute from integers alone) and writes, per case, the capped size and the
SHA-256 of the resized pixels to ``native/frame_engine/tests/golden/area.json``. The cases cover
both of OpenCV's area paths: whole ratios (2 has its own SIMD loop, 3 and 4 the plain one) and
fractional ratios (the weight tables of ``computeResizeAreaTab``).

The data is committed; the Rust test reads it. This script is deleted with the Python engine at
the switch (plan 17 N3). ``--check`` regenerates into memory and compares with the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'native' / 'frame_engine' / 'tests' / 'golden' / 'area.json'

#: (width, height, max side) of each crop.
CASES = (
    (512, 256, 256),  # ratio 2 on both sides: the SIMD loop
    (1000, 750, 500),  # ratio 2
    (66, 34, 33),  # ratio 2, widths around the vector length
    (900, 600, 300),  # ratio 3
    (1024, 1024, 256),  # ratio 4
    (1001, 2048, 1024),  # 2 on the long side, 1001 / 500 on the short one
    (1537, 1025, 512),  # fractional on both
    (640, 427, 333),
    (3000, 7, 1000),  # a thin strip: one side rounds to 2 pixels
    (5000, 1, 640),  # one row: the short side stays at 1
    (257, 513, 256),  # just above the cap
    (8, 300, 299),
    (4000, 3000, 1024),
)


def pattern(w: int, h: int) -> np.ndarray:
    """``h x w x 3`` uint8: a hash of ``(x, y, channel)``, the same arithmetic as the Rust test."""
    y, x, c = np.meshgrid(
        np.arange(h, dtype=np.uint64),
        np.arange(w, dtype=np.uint64),
        np.arange(3, dtype=np.uint64),
        indexing='ij',
    )
    mask = np.uint64(0xFFFFFFFF)
    v = (x * np.uint64(73856093)) ^ (y * np.uint64(19349663)) ^ (c * np.uint64(83492791))
    v = ((v & mask) * np.uint64(2654435761)) & mask
    out: np.ndarray = (v >> np.uint64(24)).astype(np.uint8)
    return out


def capped(pixels: np.ndarray, max_side: int) -> np.ndarray:
    """``CropJpegs._encode``'s cap, as written there (before the encode)."""
    side = max(pixels.shape[:2])
    if not 0 < max_side < side:
        return pixels
    scale = max_side / side
    h, w = pixels.shape[:2]
    return cv2.resize(
        pixels,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def goldens() -> dict[str, Any]:
    cases = []
    for w, h, max_side in CASES:
        out = capped(pattern(w, h), max_side)
        cases.append({
            'w': w,
            'h': h,
            'max_side': max_side,
            'out_w': int(out.shape[1]),
            'out_h': int(out.shape[0]),
            'sha256': hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest(),
        })  # fmt: skip
    return {
        'generated_by': 'scripts/codegen/frame_engine_area_goldens.py',
        'opencv': cv2.__version__,
        'cases': cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='compare with the committed file')
    args = parser.parse_args()
    text = json.dumps(goldens(), indent=1) + '\n'
    if args.check:
        same = OUT.read_text() == text
        print('area goldens up to date' if same else f'{OUT} is stale')
        return 0 if same else 1
    OUT.write_text(text)
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
