"""Build a labelled crop pool from public data for judge calibration (pool.jsonl + crop files).

    judge_pool.py coco   --api URL --project NAME --labels labels.jsonl --images-dir DIR --out DIR
    judge_pool.py oi     --dataset-dir DIR --out DIR      (Open Images vehicle boxes)
    judge_pool.py lfw    --dataset-dir DIR --out DIR      (face identities; local judges only)

COCO crops take their boxes from the project through the API (read only) and their labels from
the COCO annotations (``cluster_eval_set.py build``); pixels come from the local COCO copy.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from PIL import Image, ImageOps


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.cluster_eval_set import fetch_crops
from scripts.curation._project_worker_utils import curation_api_prefix, scoped_url


MIN_SIDE_PX = 48
MIN_PER_LABEL = 12


def crop_png(image: Image.Image, box: Sequence[float], *, pad: float = 0.0) -> bytes | None:
    """The normalized box (optionally padded) as PNG bytes; None when it is under MIN_SIDE_PX."""
    w, h = image.size
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    left, top = max(0, round((x1 - px) * w)), max(0, round((y1 - py) * h))
    right, bottom = min(w, round((x2 + px) * w)), min(h, round((y2 + py) * h))
    if right - left < MIN_SIDE_PX or bottom - top < MIN_SIDE_PX:
        return None
    buf = io.BytesIO()
    image.crop((left, top, right, bottom)).convert('RGB').save(buf, 'PNG')
    return buf.getvalue()


def keep_labels(rows: list[dict[str, Any]], minimum: int = MIN_PER_LABEL) -> list[dict[str, Any]]:
    counts = Counter(r['label'] for r in rows)
    return [r for r in rows if counts[r['label']] >= minimum]


def _write_pool(out: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(keep_labels(rows), key=lambda r: r['item_id'])
    (out / 'pool.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    counts = Counter(r['label'] for r in rows)
    print(f'{len(rows)} crops, {len(counts)} labels: {dict(counts.most_common(12))}')


def _save(out: Path, item_id: str, png: bytes) -> str:
    path = out / 'crops' / f'{item_id}.png'
    path.write_bytes(png)
    return str(path)


def _open(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)) or Image.open(path)


def build_coco(args: argparse.Namespace) -> None:
    out = Path(args.out)
    (out / 'crops').mkdir(parents=True, exist_ok=True)
    labels = {
        d['item_id']: d for d in (json.loads(x) for x in Path(args.labels).read_text().splitlines())
    }
    base = scoped_url(args.api.rstrip('/'), curation_api_prefix(), args.project, '')
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=60) as client:
        for crop in fetch_crops(client, base):
            lab = labels.get(crop['id'])
            src = Path(args.images_dir) / Path(crop['image_path']).name
            if lab is None or not src.is_file():
                continue
            png = crop_png(_open(src), crop['bbox_norm'])
            if png is None:
                continue
            rows.append(
                {
                    'item_id': crop['id'],
                    'label': lab['label'],
                    'group': lab['group'],
                    'bbox_norm': crop['bbox_norm'],
                    'path': _save(out, crop['id'], png),
                    'source': f'coco:{args.project}',
                }
            )
    _write_pool(out, rows)


def build_oi(args: argparse.Namespace) -> None:
    out, root = Path(args.out), Path(args.dataset_dir)
    (out / 'crops').mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for line in (root / 'labels.jsonl').read_text().splitlines():
        doc = json.loads(line)
        image_id = Path(doc['path']).stem
        img = None
        for n, b in enumerate(doc['boxes']):
            if b['area'] < 0.1:
                continue
            img = img or _open(root / doc['path'])
            png = crop_png(img, b['box'], pad=0.02)
            if png is not None:
                item = f'{image_id}_{n}'
                rows.append(
                    {
                        'item_id': item,
                        'label': b['class_name'].lower(),
                        'group': image_id,
                        'bbox_norm': b['box'],
                        'path': _save(out, item, png),
                        'source': 'openimages',
                    }
                )
    _write_pool(out, rows)


def build_lfw(args: argparse.Namespace) -> None:
    out, root = Path(args.out), Path(args.dataset_dir)
    (out / 'crops').mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for line in (root / 'labels.jsonl').read_text().splitlines():
        doc = json.loads(line)
        item = Path(doc['path']).stem
        buf = io.BytesIO()
        _open(root / doc['path']).convert('RGB').save(buf, 'PNG')
        rows.append(
            {
                'item_id': item,
                'label': doc['identity'],
                'group': item,
                'path': _save(out, item, buf.getvalue()),
                'source': 'lfw',
            }
        )
    counts = Counter(r['label'] for r in rows)
    rows = [r for r in rows if counts[r['label']] >= 6]
    _write_pool(out, rows)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    coco = sub.add_parser('coco')
    coco.add_argument('--api', required=True)
    coco.add_argument('--project', required=True)
    coco.add_argument('--labels', required=True)
    coco.add_argument('--images-dir', required=True)
    coco.set_defaults(fn=build_coco)
    for name, fn in (('oi', build_oi), ('lfw', build_lfw)):
        p = sub.add_parser(name)
        p.add_argument('--dataset-dir', required=True)
        p.set_defaults(fn=fn)
    for p in sub.choices.values():
        p.add_argument('--out', required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    args.fn(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
