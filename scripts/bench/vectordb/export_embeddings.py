"""Export real PE-Core-L14-336 embeddings of a pinned public COCO image set.

Run in the bench venv (torch, open_clip, opencv); never added to ``requirements.txt``::

    CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 venv/bin/python export_embeddings.py \\
        --images DIR/images --annotations-cache CACHE --out OUT --held-out 1000

Every pinned image yields one whole-frame vector and one vector per ground-truth box crop
(largest boxes first, at most ``--max-crops``), the same shape of workload as ingest at policy
``all``. The vectors are the model's own L2-normalized 1024-dimension image embeddings. Output
(local, gitignored): ``vectors.f32.npy`` and ``meta.parquet`` (item_id, image_id, kind,
class_name, held_out). Only public data is read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


_REPO = Path(__file__).resolve().parents[3]


def _load_preprocess() -> Any:
    """Import ``pe_preprocess`` by path: it needs only cv2 and numpy, not the ``src`` package."""
    path = _REPO / 'src' / 'services' / 'detection' / 'pe_preprocess.py'
    spec = importlib.util.spec_from_file_location('pe_preprocess', path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules['pe_preprocess'] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def load_boxes(cache: Path, image_ids: set[int]) -> tuple[dict[int, list[tuple]], dict[int, str]]:
    """COCO instance boxes per image id: (x, y, w, h, category) and the category names."""
    boxes: dict[int, list[tuple]] = {i: [] for i in image_ids}
    names: dict[int, str] = {}
    for split in ('val2017', 'train2017'):
        path = next(cache.rglob(f'instances_{split}.json'))
        data = json.loads(path.read_text())
        names.update({c['id']: c['name'] for c in data['categories']})
        for ann in data['annotations']:
            if ann['image_id'] in boxes and not ann.get('iscrowd'):
                boxes[ann['image_id']].append((*ann['bbox'], ann['category_id']))
        del data
    return boxes, names


class ImageCrops:
    """Dataset: one pinned image -> its whole-frame tensor plus the box crops, as uint8 HWC."""

    def __init__(self, paths: list[Path], boxes: dict[int, list[tuple]], args: Any, pp: Any):
        self.paths, self.boxes, self.args, self.pp = paths, boxes, args, pp

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[int, list[np.ndarray], list[int]]:
        import cv2

        path = self.paths[idx]
        image_id = int(path.stem)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return image_id, [], []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        tiles = [self.pp.resize_crop_rgb(rgb)]
        cats = [-1]
        chosen = sorted(self.boxes.get(image_id, []), key=lambda b: -b[2] * b[3])
        for x, y, bw, bh, cat in chosen:
            if len(tiles) > self.args.max_crops:
                break
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
            if min(x1 - x0, y1 - y0) < self.args.min_side:
                continue
            tiles.append(self.pp.resize_crop_rgb(np.ascontiguousarray(rgb[y0:y1, x0:x1])))
            cats.append(cat)
        return image_id, tiles, cats


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--annotations-cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--held-out', type=int, default=1000)
    ap.add_argument('--max-crops', type=int, default=8)
    ap.add_argument('--min-side', type=int, default=20)
    ap.add_argument(
        '--skip-meta',
        type=Path,
        nargs='*',
        default=[],
        help='meta.parquet of earlier exports; their images are skipped',
    )
    ap.add_argument('--limit', type=int, default=0, help='only the first N images (smoke test)')
    ap.add_argument('--seed', type=int, default=20261011)
    ap.add_argument('--batch', type=int, default=96)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--model', default='hf-hub:timm/PE-Core-L-14-336')
    args = ap.parse_args()

    import open_clip
    import pandas as pd

    pp = _load_preprocess()
    paths = sorted(args.images.glob('*.jpg'))
    if args.skip_meta:
        done_ids = {int(i) for p in args.skip_meta for i in pd.read_parquet(p)['image_id'].unique()}
        paths = [p for p in paths if int(p.stem) not in done_ids]
    if args.limit:
        paths = paths[: args.limit]
    boxes, names = load_boxes(args.annotations_cache, {int(p.stem) for p in paths})
    model, _, _ = open_clip.create_model_and_transforms(args.model)
    model = model.train(False).half().cuda()
    ds = ImageCrops(paths, boxes, args, pp)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, num_workers=args.workers, collate_fn=lambda b: b, prefetch_factor=4
    )

    vecs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    pending: list[np.ndarray] = []
    pending_meta: list[tuple[int, int]] = []
    t0 = time.perf_counter()

    def flush() -> None:
        if not pending:
            return
        x = torch.from_numpy(np.stack(pending)).cuda()
        x = (x.permute(0, 3, 1, 2).half() / 255.0 - 0.5) / 0.5
        with torch.inference_mode():
            feats = model.encode_image(x, normalize=True)
        vecs.append(feats.float().cpu().numpy())
        for image_id, cat in pending_meta:
            kind = 'whole' if cat < 0 else 'crop'
            rows.append(
                {
                    'image_id': image_id,
                    'kind': kind,
                    'class_name': names.get(cat, '_frame'),
                }
            )
        pending.clear()
        pending_meta.clear()

    done = 0
    for batch in loader:
        for image_id, tiles, cats in batch:
            for tile, cat in zip(tiles, cats, strict=True):
                pending.append(tile)
                pending_meta.append((image_id, cat))
        if len(pending) >= args.batch:
            flush()
        done += len(batch)
        if done % 400 < 8:
            rate = sum(len(v) for v in vecs) / (time.perf_counter() - t0)
            print(f'{done}/{len(ds)} images, {rate:.0f} vectors/s', flush=True)
    flush()

    matrix = np.concatenate(vecs).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    meta = pd.DataFrame(rows)
    meta.insert(0, 'item_id', np.arange(len(meta), dtype=np.int64))
    rng = np.random.default_rng(args.seed)
    image_ids = np.unique(meta['image_id'].to_numpy())
    held = set(rng.choice(image_ids, size=min(args.held_out, len(image_ids)), replace=False))
    meta['held_out'] = meta['image_id'].isin(held)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / 'vectors.f32.npy', matrix)
    meta.to_parquet(args.out / 'meta.parquet')
    print(
        json.dumps(
            {
                'vectors': len(meta),
                'images': len(image_ids),
                'vectors_per_image': len(meta) / len(image_ids),
                'dim': int(matrix.shape[1]),
                'seconds': round(time.perf_counter() - t0, 1),
                'model': args.model,
            }
        )
    )


if __name__ == '__main__':
    main()
