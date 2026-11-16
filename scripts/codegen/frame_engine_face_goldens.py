"""Goldens for the native engine's face branches (plan 17 N1b), rendered into ``faces.json`` by
``scripts/codegen/frame_engine_goldens.py``.

Runs the numpy functions the native ``faces`` and ``face_embed`` branches port
(``frame_engine/faces/scrfd.py``, ``frame_engine/faces/alignment.py`` and the vector
normalisation of ``frame_engine/branches/faces.py``) on inputs fixed here:

* ``passes``: ``plan_passes`` for photo sizes from the pinned manifests and awkward edges, under
  the plan settings a caller sends (whole frame, coarse square, tiles with overlap);
* ``decode``: random float32 detector outputs for one input, ``decode`` at a threshold, as
  ``(score, box, landmarks)`` rows;
* ``photos``: a photo's passes, one random output set per pass, then ``to_source`` per pass and
  ``merge`` (NMS over every pass, the cap counted);
* ``nms``: boxes with tied scores, identical boxes and overlaps at the threshold;
* ``align``: five landmarks under random similarities (mirrored, tiny, huge) and degenerate sets,
  ``estimate``, ``warp_params``;
* ``normalise``: float32 rows L2-normalised as ``FaceEmbed.finish`` does.

Every float32 is written as its shortest decimal (``str(np.float32)``), which a float32 parse of
the text gives back to the bit; the forward matrices are float64 written as ``repr``, as strings.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EDGE_SIZES = [(1, 1), (1, 4000), (4000, 1), (3, 7), (639, 641), (640, 640), (641, 640),
              (1280, 1281), (5472, 3648), (3648, 5472), (8000, 6000), (6000, 8000),
              (2560, 1001), (1001, 2560), (2562, 1000)]  # fmt: skip
#: Plan settings: input side, coarse square, tile side, overlap.
PASS_SETTINGS = [(640, 0, 0, 0), (640, 320, 0, 0), (640, 0, 1280, 64), (640, 320, 2560, 128),
                 (320, 160, 1920, 32), (640, 0, 640, 0), (640, 0, 1281, 0)]  # fmt: skip
# (2560, 1001) and (1001, 2560) at a tile side of 1280 scale an odd side by 0.5 (a tie for
# ``round``); (2562, 1000) at 1281 puts a tile start at 320.5.


def _f32(values: Any) -> Any:
    """float32 values as their shortest decimal (``str(np.float32)``), nested as given."""
    import numpy as np

    a = np.asarray(values, np.float32)
    return [_f32(row) for row in a] if a.ndim else float(str(a[()]))


def _sizes(rng: random.Random) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    for name in ('coco_bench_2000', 'commons_hires_500'):
        path = ROOT / 'scripts' / 'datasets' / 'manifests' / f'{name}.json'
        for image in json.loads(path.read_text(encoding='utf-8'))['images']:
            seen.add((image['width'], image['height']))
    return [*EDGE_SIZES, *rng.sample(sorted(seen), 60)]


def _passes(sizes: list[tuple[int, int]]) -> list[dict[str, Any]]:
    from frame_engine.faces import scrfd

    rows = []
    for w, h in sizes:
        for net, coarse, tile_side, overlap in PASS_SETTINGS:
            got = scrfd.plan_passes(w, h, net, coarse=coarse, tile_side=tile_side, overlap=overlap)
            rows.append({
                'size': [w, h], 'settings': [net, coarse, tile_side, overlap],
                'passes': [[*p.scaled, *p.window, *(int(b) for b in p.inner)] for p in got],
            })  # fmt: skip
    return rows


def _outputs(rng: random.Random, net: int, fire: float) -> dict[str, list[Any]]:
    """One input's nine outputs: scores mostly low, a share ``fire`` above 0.5, some exactly at
    it; distances and landmark offsets in strides."""
    import numpy as np

    from frame_engine.faces import scrfd

    out: dict[str, list[Any]] = {}
    for s in scrfd.FPN_STRIDES:
        n = (net // s) ** 2 * scrfd.NUM_ANCHORS
        scores = [rng.uniform(0.0, 0.49) if rng.random() > fire else rng.uniform(0.5, 1.0)
                  for _ in range(n)]  # fmt: skip
        for k in rng.sample(range(n), max(1, n // 40)):
            scores[k] = 0.5
        out[f'score_{s}'] = _f32(scores)
        out[f'bbox_{s}'] = _f32(np.array([rng.uniform(0.0, 4.0) for _ in range(n * 4)]))
        out[f'kps_{s}'] = _f32(np.array([rng.uniform(-3.0, 3.0) for _ in range(n * 10)]))
    return out


def _arrays(outputs: dict[str, list[Any]]) -> dict[str, Any]:
    import numpy as np

    return {k: np.asarray(v, np.float32).reshape(-1, 1 if k.startswith('score') else
                                                 4 if k.startswith('bbox') else 10)
            for k, v in outputs.items()}  # fmt: skip


def _detections(scores: Any, boxes: Any, kps: Any) -> dict[str, Any]:
    return {'scores': _f32(scores), 'boxes': _f32(boxes).copy() if len(boxes) else [],
            'landmarks': _f32(kps.reshape(-1, 10)) if len(kps) else []}  # fmt: skip


def _decode(rng: random.Random) -> list[dict[str, Any]]:
    from frame_engine.faces import scrfd

    cases = []
    for net, threshold, fire in ((64, 0.5, 0.1), (64, 0.75, 0.3), (32, 0.5, 0.0)):
        outputs = _outputs(rng, net, fire)
        got = scrfd.decode(_arrays(outputs), net, net, threshold)
        cases.append({'net': net, 'threshold': threshold, 'outputs': outputs,
                      'want': _detections(*got)})  # fmt: skip
    return cases


def _photos(rng: random.Random) -> list[dict[str, Any]]:
    from frame_engine.faces import scrfd

    cases = []
    for (w, h), (net, coarse, tile_side, overlap), nms, cap in (
        ((1500, 1000), (64, 32, 128, 16), 0.4, 1000),
        ((700, 1900), (64, 0, 160, 8), 0.4, 6),
        ((90, 60), (64, 32, 0, 0), 0.3, 50),
    ):
        passes = scrfd.plan_passes(w, h, net, coarse=coarse, tile_side=tile_side, overlap=overlap)
        outputs = [_outputs(rng, net, 0.08) for _ in passes]
        parts = [scrfd.to_source(p, w, h, *scrfd.decode(_arrays(o), net, net, 0.5))
                 for p, o in zip(passes, outputs, strict=True)]  # fmt: skip
        faces = scrfd.merge(parts, nms, cap)
        cases.append({
            'size': [w, h], 'settings': [net, coarse, tile_side, overlap], 'nms': nms, 'cap': cap,
            'outputs': outputs, 'to_source': [_detections(*p) for p in parts],
            'want': {**_detections(faces.scores, faces.boxes, faces.landmarks),
                     'overflow': faces.overflow},
        })  # fmt: skip
    return cases


def _nms() -> list[dict[str, Any]]:
    import numpy as np

    from frame_engine.faces import scrfd

    cases = []
    boxes = [[0, 0, 9, 9], [0, 0, 9, 9], [1, 1, 10, 10], [20, 20, 30, 30], [20, 20, 30, 30],
             [0, 0, 9, 9], [5, 0, 14, 9], [100, 100, 100, 100], [0.5, 0.25, 9.75, 9.5]]  # fmt: skip
    scores = [0.9, 0.9, 0.95, 0.6, 0.6, 0.9, 0.9, 0.7, 0.95]
    for threshold in (0.0, 0.3333333, 0.4, 0.5, 1.0):
        keep = scrfd.nms(np.array(boxes, np.float32), np.array(scores, np.float32), threshold)
        cases.append({'boxes': _f32(boxes), 'scores': _f32(scores), 'threshold': threshold,
                      'keep': keep.tolist()})  # fmt: skip
    # IoU at the threshold itself: two 10 x 10 boxes (``+1`` areas 100) sharing 50 pixels
    pair = [[0, 0, 9, 9], [5, 0, 14, 9]]
    for threshold in (1.0 / 3.0, 0.33333334, 0.3333333):
        keep = scrfd.nms(np.array(pair, np.float32), np.array([0.8, 0.8], np.float32), threshold)
        cases.append({'boxes': _f32(pair), 'scores': _f32([0.8, 0.8]), 'threshold': threshold,
                      'keep': keep.tolist()})  # fmt: skip
    return cases


def _landmarks(rng: random.Random) -> list[list[list[float]]]:
    from frame_engine.faces import alignment

    sets = []
    for k in range(120):
        scale = math.exp(rng.uniform(math.log(0.2), math.log(40.0)))
        angle = rng.uniform(-1.2, 1.2)
        mirror = -1.0 if k % 9 == 0 else 1.0
        tx, ty = rng.uniform(-50, 6000), rng.uniform(-50, 4000)
        c, s = math.cos(angle) * scale, math.sin(angle) * scale
        pts = []
        for dx, dy in alignment.ARCFACE_DST.tolist():
            x, y = mirror * (dx - 56.0), dy - 56.0
            pts.append([c * x - s * y + tx + rng.uniform(-2, 2) * scale / 4,
                        s * x + c * y + ty + rng.uniform(-2, 2) * scale / 4])  # fmt: skip
        sets.append(pts)
    # exact similarities: the span lands a few ulps either side of a whole number of taps
    for scale in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        for angle in (0.0, 0.3, -0.5):
            c, s = math.cos(angle) * scale, math.sin(angle) * scale
            sets.append([[c * x - s * y + 11.0, s * x + c * y + 7.0]
                         for x, y in alignment.ARCFACE_DST.tolist()])  # fmt: skip
    sets.append([[10.0, 10.0]] * 5)
    sets.append([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    sets.append([[0.0, 5.0], [10.0, 5.0], [20.0, 5.0], [30.0, 5.0], [40.0, 5.0]])
    sets.append([[100.0, 100.0], [100.0001, 100.0], [100.0, 100.0001], [100.0, 100.0],
                 [100.0001, 100.0001]])  # fmt: skip
    return sets


def _align(rng: random.Random) -> list[dict[str, Any]]:
    import numpy as np

    from frame_engine.faces import alignment

    rows = []
    for pts in _landmarks(rng):
        lmk = np.asarray(pts, np.float32)
        inv, taps, ok = alignment.warp_params(lmk[None])
        fwd = alignment.estimate(lmk.astype(np.float64))
        rows.append({'landmarks': _f32(lmk), 'forward': [repr(float(v)) for v in fwd.ravel()],
                     'inverse': _f32(inv[0]), 'taps': int(taps[0]), 'ok': bool(ok[0])})  # fmt: skip
    return rows


def _normalise(rng: random.Random) -> list[dict[str, Any]]:
    import numpy as np

    rows = []
    for n, dim in ((3, 512), (2, 5), (2, 8), (2, 9), (2, 100), (2, 129), (2, 300), (1, 1000)):
        v = np.array([[rng.gauss(0.0, rng.choice((0.01, 1.0, 30.0))) for _ in range(dim)]
                      for _ in range(n)], np.float32)  # fmt: skip
        if dim == 8:
            v[1] = 0.0
        norms = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-10)
        rows.append({'vectors': _f32(v), 'unit': _f32(v / norms)})
    return rows


def render() -> dict[str, Any]:
    """The ``faces.json`` document."""
    rng = random.Random(20261116)
    return {
        'generated_by': 'scripts/codegen/frame_engine_face_goldens.py',
        'sources': ['frame_engine/faces/scrfd.py', 'frame_engine/faces/alignment.py',
                    'frame_engine/branches/faces.py'],
        'passes': _passes(_sizes(rng)),
        'decode': _decode(rng),
        'photos': _photos(rng),
        'nms': _nms(),
        'align': _align(rng),
        'normalise': _normalise(rng),
    }  # fmt: skip
