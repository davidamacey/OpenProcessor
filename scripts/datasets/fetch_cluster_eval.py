#!/usr/bin/env python3
"""Fetch the pinned 3,000-item cluster evaluation image set (#341, public data only).

Selects images from COCO val2017 and Open Images v7 validation + test (seed 20261028, see
``cluster_eval_selection.py``), downloads them, makes the near-duplicate variants (seeded crop
jitter and JPEG re-encode of 100 sources), and writes next to ``images/``: ``coco_gt.json`` (the
datasets' own boxes in COCO form, variants included), ``ATTRIBUTION.csv`` and the pin.

    python scripts/datasets/fetch_cluster_eval.py --out <dir> --cache-dir <cache> \\
        --manifest scripts/datasets/manifests/cluster_eval_3000.json
    python scripts/datasets/fetch_cluster_eval.py --verify-only --out <dir> \\
        --manifest scripts/datasets/manifests/cluster_eval_3000.json

With an existing ``--manifest`` the selection is recomputed and must equal the pin (a drift fails
the run); without one the selection is written as the new pin. ``--verify-only`` re-hashes the
base images against the pin and needs no network. Near-duplicate variants depend on the Pillow
and libjpeg build, so their bytes are recorded but only their presence is verified.

The images stay out of git; only the pin (ids, licences, authors, source URLs, hashes) is committed.
Run it on the host, not in a container.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.datasets import cluster_eval_selection as sel
from scripts.datasets._common import FetchError, download, sha256_file, write_csv, write_json
from scripts.datasets.coco_fetch import IMAGE_URL_TMPL, ensure_annotations
from scripts.datasets.coco_selection import DEFAULT_LICENSES, resolve_license_names
from scripts.datasets.fetch_openimages_plates import (
    ALLOWED_LICENSE_URL,
    BBOX_URL,
    CLASS_DESC_URL,
    IMAGE_DOWNLOAD_TMPL,
    IMAGES_URL,
    fetch_csv_text,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('fetch_cluster_eval')

OI_SPLITS = ('validation', 'test')
WORKERS = 8


def oi_class_ids(class_desc_csv: str) -> dict[str, str]:
    """``LabelName -> evaluation class name`` for the Open Images classes the set uses."""
    by_display = {
        r['DisplayName']: r['LabelName'] for r in csv.DictReader(io.StringIO(class_desc_csv))
    }
    missing = [n for n in sel.OI_CLASSES if n not in by_display]
    if missing:
        raise FetchError(f'Open Images class names not found: {missing}')
    return {by_display[n]: label for n, label in sel.OI_CLASSES.items()}


def oi_boxes(bbox_csv: str, mids: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    """Confident, single-object, non-depiction boxes of the used classes per ImageID."""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in csv.DictReader(io.StringIO(bbox_csv)):
        name = mids.get(r['LabelName'])
        if name is None or r['IsGroupOf'] == '1' or r['IsDepiction'] == '1':
            continue
        out.setdefault(r['ImageID'], []).append(
            {
                'name': name,
                'x_min': float(r['XMin']),
                'x_max': float(r['XMax']),
                'y_min': float(r['YMin']),
                'y_max': float(r['YMax']),
            }
        )
    return out


def oi_metadata(images_csv: str, split: str) -> dict[str, dict[str, Any]]:
    """CC BY 2.0 images without a stored rotation (the pixels are used as they are)."""
    return {
        r['ImageID']: {
            'split': split,
            'license': r['License'],
            'author': r.get('Author', ''),
            'original_url': r.get('OriginalURL', ''),
        }
        for r in csv.DictReader(io.StringIO(images_csv))
        if r['License'] == ALLOWED_LICENSE_URL and r.get('Rotation', '') in ('', '0', '0.0')
    }


def _fetch_one(row: dict[str, Any], images_dir: Path) -> None:
    if row['source'] == 'coco':
        url = IMAGE_URL_TMPL[row['split']].format(file_name=row['file_name'])
    else:
        url = IMAGE_DOWNLOAD_TMPL.format(split=row['split'], image_id=row['image_id'])
    dest = download(url, images_dir / row['file_name'])
    with Image.open(dest) as im:
        row['width'], row['height'] = im.size
    row['bytes'] = dest.stat().st_size
    row['sha256'] = sha256_file(dest)


def _make_variant(parent: Path, plan: dict[str, Any], dest: Path) -> dict[str, Any]:
    with Image.open(parent) as im:
        w, h = im.size
        window = sel.jitter_window(w, h, plan['margins'])
        im.convert('RGB').crop(window).save(dest, 'JPEG', quality=plan['quality'])
    with Image.open(dest) as out:
        width, height = out.size
    return {
        **plan,
        'window': list(window),
        'width': width,
        'height': height,
        'bytes': dest.stat().st_size,
        'sha256': sha256_file(dest),
    }


def _gt(
    base: list[dict[str, Any]],
    near_dups: list[dict[str, Any]],
    coco_val: dict[str, Any],
    oi: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    coco_names = {c['id']: c['name'] for c in coco_val['categories']}
    coco_by_image: dict[int, list[dict[str, Any]]] = {}
    for a in coco_val['annotations']:
        coco_by_image.setdefault(a['image_id'], []).append(a)
    names = sorted(set(coco_names.values()) | set(sel.OI_CLASSES.values()))
    cat_id = {n: i for i, n in enumerate(names, 1)}
    images, anns = [], []
    per_file: dict[str, list[dict[str, Any]]] = {}
    for n, r in enumerate(base, 1):
        gid = r['image_id'] if r['source'] == 'coco' else 10_000_000 + n
        images.append(
            {'id': gid, 'file_name': r['file_name'], 'width': r['width'], 'height': r['height']}
        )
        rows: list[dict[str, Any]] = []
        if r['source'] == 'coco':
            rows = [
                {'bbox': a['bbox'], 'iscrowd': a['iscrowd'], 'name': coco_names[a['category_id']]}
                for a in coco_by_image.get(r['image_id'], [])
            ]
        else:
            w, h = r['width'], r['height']
            rows = [
                {
                    'bbox': [
                        b['x_min'] * w,
                        b['y_min'] * h,
                        (b['x_max'] - b['x_min']) * w,
                        (b['y_max'] - b['y_min']) * h,
                    ],
                    'iscrowd': 0,
                    'name': b['name'],
                }
                for b in oi.get(r['image_id'], [])
            ]
        per_file[r['file_name']] = rows
    gids = {i['file_name']: i['id'] for i in images}
    nxt = 20_000_000
    for nd in near_dups:
        for v in nd['variants']:
            nxt += 1
            gids[v['file_name']] = nxt
            images.append(
                {
                    'id': nxt,
                    'file_name': v['file_name'],
                    'width': v['width'],
                    'height': v['height'],
                    'near_duplicate_of': nd['parent'],
                }
            )
            per_file[v['file_name']] = sel.shift_annotations(
                per_file[nd['parent']],
                tuple(v['window']),
            )
    k = 0
    for fname, rows in per_file.items():
        for a in rows:
            k += 1
            anns.append(
                {
                    'id': k,
                    'image_id': gids[fname],
                    'category_id': cat_id[a['name']],
                    'bbox': [round(float(x), 2) for x in a['bbox']],
                    'iscrowd': a['iscrowd'],
                }
            )
    return {
        'images': images,
        'annotations': anns,
        'categories': [{'id': i, 'name': n} for n, i in cat_id.items()],
    }


def verify(out: Path, pin: dict[str, Any]) -> list[str]:
    problems = []
    for r in pin['images']:
        p = out / 'images' / r['file_name']
        if not p.is_file():
            problems.append(f'{r["file_name"]}: missing')
        elif sha256_file(p) != r['sha256']:
            problems.append(f'{r["file_name"]}: sha256 differs from the pin')
    for nd in pin['near_duplicates']:
        problems += [
            f'{v["file_name"]}: missing'
            for v in nd['variants']
            if not (out / 'images' / v['file_name']).is_file()
        ]
    return problems


def select(cache_dir: Path, seed: int, scale: float) -> dict[str, Any]:
    val, train = ensure_annotations(cache_dir)
    rare = sel.rarest_coco_classes(train)
    del train
    coco_rows = sel.coco_images(val, resolve_license_names(list(DEFAULT_LICENSES)))
    mids = oi_class_ids(fetch_csv_text(CLASS_DESC_URL, cache_dir, 'oi_class_descriptions.csv'))
    boxes: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for split in OI_SPLITS:
        boxes |= oi_boxes(fetch_csv_text(BBOX_URL[split], cache_dir, f'oi_{split}_bbox.csv'), mids)
        meta |= oi_metadata(
            fetch_csv_text(IMAGES_URL[split], cache_dir, f'oi_{split}_images.csv'), split
        )
    oi_rows = sel.openimages_images(boxes, meta)
    base = sel.select_base(coco_rows, oi_rows, {r['class'] for r in rare}, seed, scale)
    return {
        'val': val,
        'rare': rare,
        'base': base,
        'boxes': boxes,
        'near_dups': sel.select_near_duplicates(base, seed),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    chosen = select(args.cache_dir, args.seed, args.scale)
    base, near_dups = chosen['base'], chosen['near_dups']
    pinned = json.loads(args.manifest.read_text()) if args.manifest.is_file() else None
    if pinned is not None:
        got = [r['file_name'] for r in base]
        if got != [r['file_name'] for r in pinned['images']] or [
            (n['parent'], [v['file_name'] for v in n['variants']]) for n in near_dups
        ] != [
            (n['parent'], [v['file_name'] for v in n['variants']])
            for n in pinned['near_duplicates']
        ]:
            raise FetchError(f'{args.manifest}: the selection no longer equals the pin')
    images_dir = args.out / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda r: _fetch_one(r, images_dir), base))
    for nd in near_dups:
        nd['variants'] = [
            _make_variant(images_dir / nd['parent'], v, images_dir / v['file_name'])
            for v in nd['variants']
        ]
    gt = _gt(base, near_dups, chosen['val'], chosen['boxes'])
    write_json(args.out / 'coco_gt.json', gt)
    ann_sha = sha256_file(args.out / 'coco_gt.json')
    pin = sel.build_pin(base, near_dups, chosen['rare'], args.seed, args.scale, ann_sha)
    if pinned is not None:
        bad = [
            r['file_name']
            for r, p in zip(base, pinned['images'], strict=True)
            if r['sha256'] != p['sha256']
        ]
        if bad or pinned['annotations_sha256'] != ann_sha:
            raise FetchError(f'{args.manifest}: {len(bad)} image hashes or the annotations differ')
    else:
        write_json(args.manifest, pin)
    write_csv(
        args.out / 'ATTRIBUTION.csv',
        base,
        ('file_name', 'source', 'image_id', 'license', 'author', 'attribution_url'),
        extrasaction='ignore',
    )
    return pin


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, default=Path('cache/datasets'))
    p.add_argument('--seed', type=int, default=sel.SEED)
    p.add_argument('--scale', type=float, default=1.0, help='multiplier on the per-group quotas')
    p.add_argument('--verify-only', action='store_true')
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_only:
            problems = verify(args.out, json.loads(args.manifest.read_text()))
            for line in problems:
                logger.error(line)
            return 1 if problems else 0
        pin = run(args)
    except FetchError as exc:
        logger.error('%s', exc)
        return 1
    logger.info('pinned %d images, manifest_sha256 %s', len(pin['images']), pin['manifest_sha256'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
