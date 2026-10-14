"""Paths, image and dataset builders and a scripted detector shared by the bake-off harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.curation.bakeoff.backends.base import Detection


REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS = REPO_ROOT / 'scripts/curation/bakeoff'


def _img(path: Path, w: int = 100, h: int = 50) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))
    return path


def _mc_dataset(
    root: Path,
    labels: dict[str, list[tuple[int, float, float, float, float]]],
    names: list[str],
    *,
    fill: dict[str, int] | None = None,
) -> Path:
    """A tiny YOLO test split: 100x100 images, ``labels[stem]`` rows, ``data.yaml`` names."""
    for stem, rows in labels.items():
        img = root / 'images/test' / f'{stem}.png'
        img.parent.mkdir(parents=True, exist_ok=True)
        value = (fill or {}).get(stem, 0)
        cv2.imwrite(str(img), np.full((100, 100, 3), value, dtype=np.uint8))
        lbl = root / 'labels/test' / f'{stem}.txt'
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text(''.join(f'{c} {cx} {cy} {w} {h}\n' for c, cx, cy, w, h in rows))
    (root / 'data.yaml').write_text(
        f'path: {root}\ntest: images/test\nnc: {len(names)}\nnames: {json.dumps(names)}\n'
    )
    return root


# GT box of every _mc_dataset row below, in absolute pixels: (20, 20, 60, 60).
_BOX = (0.4, 0.4, 0.4, 0.4)


def _det(
    cls: int, score: float = 0.9, box: tuple[float, float, float, float] = (20, 20, 60, 60)
) -> Detection:
    x1, y1, x2, y2 = (float(v) for v in box)
    return Detection(x1, y1, x2, y2, score, class_id=cls)


class _FakeDetector:
    """Scripted detector: 'cat'/'dog' named classes; answers by image brightness."""

    runtime = 'fake'
    class_names: dict[int, str] | None = {0: 'cat', 1: 'dog'}

    def __init__(self, name: str) -> None:
        self.name = name

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        mean = float(image_rgb.mean())
        if mean < 50:  # image 'a': a dog -> model class 1
            return [_det(1, 0.9)]
        if mean > 150:  # image 'b': a cat -> model class 0
            return [_det(0, 0.8)]
        return []
