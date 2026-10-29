"""Embedding extraction for the fine-grained benchmark. Run in the research venv, on a locked GPU.

Every image is decoded once (a DataLoader worker, EXIF upright), its labelled box is found
(set C: the largest car or truck box of a YOLO11s detector; distractors: the stored product
box), and every requested (model, variant) square is cut from that one frame.

    flock $LOCKS/gpu2.lock nice -n 10 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
        $VENV/bin/python -m scripts.research.fine_grained.embed_run extract --out DIR --models ...

FG_SETC_ROOT (set C directory), FG_Z2_BUNDLE (the Z2 vector bundle) and FG_DETECTOR (a YOLO11s
weights file) are read from the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from scripts.research.fine_grained import data, setc
from scripts.research.fine_grained.preprocess import VARIANTS, variant_input


COCO_CAR, COCO_TRUCK = 2, 7
BATCH = 32


def build_rows(
    subset: str, distractor_cap: int | None, setc_cap: int | None = None
) -> list[dict[str, Any]]:
    root = data.env_path('FG_SETC_ROOT')
    z2_bundle = data.env_path('FG_Z2_BUNDLE')
    z2_meta, z2_labels = z2_bundle / 'vecs.meta.jsonl', z2_bundle / 'eval' / 'labels.jsonl'
    rows: list[dict[str, Any]] = [
        {
            'id': it.path,
            'source': 'setc',
            'path': str(root / it.path),
            'make': it.make,
            'model': it.model,
            'generation': it.generation,
            'box': None,
        }
        for it in setc.load_items(root)[:: max(1, 2943 // setc_cap) if setc_cap else 1]
    ]
    if subset == 'z2':
        labels = {
            r['item_id']: r for r in (json.loads(x) for x in z2_labels.read_text().splitlines())
        }
        z2 = [json.loads(x) for x in z2_meta.read_text().splitlines()]
        return [
            {
                'id': r['crop_id'],
                'source': 'z2',
                'path': r['source_identifier'],
                'make': None,
                'model': None,
                'generation': None,
                'label': labels[r['crop_id']]['label'],
                'group': labels[r['crop_id']]['group'],
                'box': r['bbox_norm'],
            }
            for r in sorted(z2, key=lambda r: r['crop_id'])
            if r['crop_id'] in labels
        ]
    if subset == 'all':
        z2 = [json.loads(x) for x in z2_meta.read_text().splitlines()]
        car = sorted(
            (r for r in z2 if r['detector_class_name'] == 'car'), key=lambda r: r['crop_id']
        )
        rows += [
            {
                'id': r['crop_id'],
                'source': 'distractor',
                'path': r['source_identifier'],
                'make': None,
                'model': None,
                'generation': None,
                'box': r['bbox_norm'],
            }
            for r in car[:distractor_cap]
        ]
    return rows


class Frames:
    """Dataset of unique image paths: decoded once, upright, RGB uint8."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[int, np.ndarray]:
        with Image.open(self.paths[i]) as im:
            return i, np.asarray(ImageOps.exif_transpose(im).convert('RGB'))


def _collate(b: list[tuple[int, np.ndarray]]) -> list[tuple[int, np.ndarray]]:
    return b


def detect_largest(model: Any, frames: list[np.ndarray]) -> list[list[float] | None]:
    """Largest car or truck box (pixel xyxy plus class and confidence) per frame, or None."""
    out: list[list[float] | None] = []
    for res in model.predict(frames, classes=[COCO_CAR, COCO_TRUCK], conf=0.25, verbose=False):
        best = None
        for xyxy, cls, conf in zip(
            res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist(), strict=True
        ):
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if best is None or area > best[0]:
                best = (area, [*xyxy, cls, conf])
        out.append(best[1] if best else None)
    return out


class Extractor:
    """One pass over the images: decode once, find the box, cut every (model, variant) square."""

    def __init__(self, args: argparse.Namespace) -> None:
        from scripts.research.fine_grained import models as enc

        self.enc = enc
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows = build_rows(args.subset, args.distractor_cap, args.setc_cap)
        self.by_path: dict[str, list[int]] = {}
        for i, r in enumerate(self.rows):
            self.by_path.setdefault(r['path'], []).append(i)
        self.paths = sorted(self.by_path)
        self.boxes_file = self.out / 'boxes_setc.json'
        self.known: dict[str, list[float] | None] = (
            json.loads(self.boxes_file.read_text()) if self.boxes_file.exists() else {}
        )
        self.encoders = {k: enc.load(k) for k in args.models}
        self.acc: dict[tuple[str, str], np.ndarray] = {}
        self.sums = {(m, v): 0.0 for m in args.models for v in args.variants}
        self.pending: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
        self.detector = self._detector()

    def _detector(self) -> Any:
        from ultralytics import YOLO

        todo = any(r['source'] == 'setc' and r['id'] not in self.known for r in self.rows)
        return YOLO(str(data.env_path('FG_DETECTOR'))) if todo else None

    def _flush(self, key: tuple[str, str], *, force: bool) -> None:
        import torch

        buf = self.pending.get(key, [])
        while len(buf) >= BATCH or (force and buf):
            chunk, buf[:] = buf[:BATCH], buf[BATCH:]
            batch = np.stack([c[1] for c in chunk])
            torch.cuda.synchronize()
            ts = time.time()
            vec = self.encoders[key[0]].encode(batch)
            torch.cuda.synchronize()
            self.sums[key] += time.time() - ts
            arr = self.acc.setdefault(key, np.zeros((len(self.rows), vec.shape[1]), np.float32))
            arr[[c[0] for c in chunk]] = vec

    def _detect(self, batch: list[tuple[int, np.ndarray]]) -> None:
        if self.detector is None:
            return
        need = [
            k
            for k, (pi, _) in enumerate(batch)
            if any(
                self.rows[i]['source'] == 'setc' and self.rows[i]['id'] not in self.known
                for i in self.by_path[self.paths[pi]]
            )
        ]
        if not need:
            return
        boxes = detect_largest(self.detector, [batch[k][1] for k in need])
        for k, box in zip(need, boxes, strict=True):
            for i in self.by_path[self.paths[batch[k][0]]]:
                self.known[self.rows[i]['id']] = box

    def _box(self, row: dict[str, Any], w: int, h: int) -> list[float] | None:
        if row['source'] == 'setc':
            b = self.known.get(row['id'])
            return b[:4] if b else None
        x1, y1, x2, y2 = row['box']
        return [x1 * w, y1 * h, x2 * w, y2 * h]

    def _queue(self, pi: int, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        img = Image.fromarray(frame)
        for i in self.by_path[self.paths[pi]]:
            box = self._box(self.rows[i], w, h)
            for m in self.args.models:
                for v in self.args.variants:
                    sq = variant_input(img, box, v, self.enc.SPECS[m].size)
                    self.pending.setdefault((m, v), []).append((i, sq))

    def run(self) -> dict[str, Any]:
        import torch
        from torch.utils.data import DataLoader

        t0 = time.time()
        loader: DataLoader[Any] = DataLoader(
            Frames(self.paths),  # type: ignore[arg-type]  # map-style dataset, not a Dataset subclass
            batch_size=16,
            num_workers=self.args.workers,
            collate_fn=_collate,
        )
        for batch in loader:
            self._detect(batch)
            for pi, frame in batch:
                self._queue(pi, frame)
            for key in list(self.pending):
                self._flush(key, force=False)
        for key in list(self.pending):
            self._flush(key, force=True)
        return self._write(t0, torch.__version__)

    def _write(self, t0: float, torch_version: str) -> dict[str, Any]:
        a = self.args
        self.boxes_file.write_text(json.dumps(self.known, sort_keys=True))
        for (m, v), arr in self.acc.items():
            np.save(self.out / f'{self.enc.SPECS[m].name}__{v}__{a.subset}.npy', arr)
        meta = {
            'git_sha': data.git_sha(),
            'load_avg': os.getloadavg(),
            'subset': a.subset,
            'models': a.models,
            'variants': a.variants,
            'rows': len(self.rows),
            'wall_s': time.time() - t0,
            'encode_s': {f'{m}__{v}': s for (m, v), s in self.sums.items()},
            'setc_no_box': sum(
                1 for r in self.rows if r['source'] == 'setc' and not self.known.get(r['id'])
            ),
            'torch': torch_version,
            'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),
        }
        (self.out / f'extract_{a.subset}_{"_".join(a.models)}.json').write_text(
            json.dumps(meta, indent=1)
        )
        lines = (json.dumps({k: v for k, v in r.items() if k != 'path'}) for r in self.rows)
        (self.out / f'meta_{a.subset}.jsonl').write_text('\n'.join(lines) + '\n')
        return meta


def extract(args: argparse.Namespace) -> None:
    print(json.dumps(Extractor(args).run(), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--out', required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--variants', nargs='+', default=['centre'], choices=VARIANTS)
    p.add_argument('--subset', choices=['setc', 'all', 'z2'], default='all')
    p.add_argument('--distractor-cap', type=int, default=None)
    p.add_argument('--setc-cap', type=int, default=None, help='smoke tests only')
    p.add_argument('--workers', type=int, default=4)
    p.set_defaults(fn=extract)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
