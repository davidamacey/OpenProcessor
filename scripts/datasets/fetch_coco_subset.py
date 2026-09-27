#!/usr/bin/env python3
"""Fetch a license-filtered, pinned COCO 2017 subset (G-05, runbook §3.3).

Builds the main OpenProcessor sample dataset: a seeded, per-class-balanced
selection of COCO 2017 images across 10 vehicle + animal classes, filtered
to Flickr licenses that are safe to redistribute crops/thumbnails of
(Attribution, Attribution-ShareAlike, "No known copyright restrictions",
"United States Government Work" -- explicitly excluding NonCommercial and
NoDerivs variants).

Pipeline:

1. Download ``annotations_trainval2017.zip`` once to ``--cache-dir``
   (default ``cache/datasets/``), verified by SHA-256.
2. Parse ``instances_val2017.json`` + ``instances_train2017.json``.
3. For each image, the *primary class* is whichever of the 10 target
   classes has the largest box (by area) on that image, restricted to
   boxes covering >= 1% of the frame.
4. Keep only images whose license (resolved by name, not hardcoded id --
   COCO's license ids are dataset-file-local) is in the allowed set.
5. Sample ``--per-class`` images per class, deterministically
   (``--seed``): val2017 first, topped up from train2017.
6. Draw disjoint side sets (``--side-sets upload=12,post_promote=24``)
   from the remaining eligible pool.
7. Verify the selection against a committed pinned manifest
   (``scripts/datasets/manifests/<name>.json``) if one exists, else write
   it as the new pin.
8. Download every selected image and write ``ATTRIBUTION.csv``,
   ``coco_gt.json`` (a COCO-format ground-truth subset), and
   ``SELECTION.json`` alongside ``images/``.

Run this on the **host** (e.g. via ``make sample-coco`` /
``make sample-coco-readme``, or directly with the project venv), never
via ``docker compose exec``. ``--out`` is resolved relative to the
current working directory, and ``docker-compose.yml`` mounts
``${OP_SOURCE_ROOT_HOST:-./data/source}`` into the ``yolo-api`` /
``curation-detection-worker`` containers **read-only** at
``/data/source``: a container path under that mount raises
``OSError: Read-only filesystem``. The host filesystem itself has no
such restriction, so ``--out data/source/<name>`` on the host, or the
default ``--out data/samples/<name>`` (see "Mounting your image source"
in ``docs/CURATION.md`` for pointing ``OP_SOURCE_ROOT_HOST`` at
``data/samples`` instead), both work.

Usage::

    python scripts/datasets/fetch_coco_subset.py --out data/samples/coco_va \\
        --per-class 80 --seed 20260925 --side-sets upload=12,post_promote=24 \\
        --manifest scripts/datasets/manifests/coco_va_800.json

    # README quick-start sample (200 images, no side sets)
    python scripts/datasets/fetch_coco_subset.py --out data/samples/coco_va_readme \\
        --n 200 --seed 20260925 \\
        --manifest scripts/datasets/manifests/coco_va_200.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.datasets._common import (
    FetchError,
    download,
    load_pinned_manifest,
    seeded_sample,
    sha256_file,
    write_csv,
    write_json,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('fetch_coco_subset')

ANNOTATIONS_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
# Verified by direct download 2026-09-25 (runbook §3.1: 252,907,541 bytes).
# COCO's file at this URL has been stable since the 2017 release.
ANNOTATIONS_SHA256 = '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'
ANNOTATIONS_SIZE = 252_907_541

DEFAULT_CLASSES = (
    'bicycle',
    'car',
    'motorcycle',
    'bus',
    'truck',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
)

# CLI --licenses token -> COCO licenses[].name (resolved by name, not id --
# see module docstring point 4).
LICENSE_ALIASES = {
    'by': 'Attribution License',
    'by-sa': 'Attribution-ShareAlike License',
    'no-known': 'No known copyright restrictions',
    'usgov': 'United States Government Work',
}
DEFAULT_LICENSES = ('by', 'by-sa', 'no-known', 'usgov')

MIN_BOX_AREA_FRACTION = 0.01
IMAGE_URL_TMPL = {
    'val2017': 'http://images.cocodataset.org/val2017/{file_name}',
    'train2017': 'http://images.cocodataset.org/train2017/{file_name}',
}


# =============================================================================
# Pure selection logic (unit-testable without network)
# =============================================================================


def resolve_license_names(tokens: list[str]) -> set[str]:
    bad = [t for t in tokens if t not in LICENSE_ALIASES]
    if bad:
        raise FetchError(f'unknown --licenses token(s) {bad}; valid: {sorted(LICENSE_ALIASES)}')
    return {LICENSE_ALIASES[t] for t in tokens}


def primary_class_per_image(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    category_id_to_name: dict[int, str],
    target_class_ids: set[int],
) -> dict[int, str]:
    """``image_id -> primary class name`` for images with a qualifying box.

    The primary class is whichever target-class box has the largest area,
    restricted to boxes covering >= ``MIN_BOX_AREA_FRACTION`` of the frame.
    Images with no qualifying box are absent from the result.
    """
    dims = {im['id']: (im['width'], im['height']) for im in images}
    best_area: dict[int, float] = {}
    best_cat: dict[int, int] = {}
    for ann in annotations:
        if ann.get('iscrowd', 0):
            continue
        cat_id = ann['category_id']
        if cat_id not in target_class_ids:
            continue
        image_id = ann['image_id']
        w, h = dims.get(image_id, (0, 0))
        frame_area = w * h
        if frame_area <= 0:
            continue
        _, _, bw, bh = ann['bbox']
        area = bw * bh
        if area / frame_area < MIN_BOX_AREA_FRACTION:
            continue
        if area > best_area.get(image_id, -1.0):
            best_area[image_id] = area
            best_cat[image_id] = cat_id
    return {img_id: category_id_to_name[cat_id] for img_id, cat_id in best_cat.items()}


def eligible_images(
    images: list[dict[str, Any]],
    primary_class: dict[int, str],
    license_id_to_name: dict[int, str],
    allowed_license_names: set[str],
    split: str,
) -> list[dict[str, Any]]:
    """Images with a primary class and an allowed license, sorted by id
    for deterministic downstream sampling."""
    out = []
    for im in images:
        cls = primary_class.get(im['id'])
        if cls is None:
            continue
        lic_name = license_id_to_name.get(im['license'])
        if lic_name not in allowed_license_names:
            continue
        out.append(
            {
                'image_id': im['id'],
                'file_name': im['file_name'],
                'split': split,
                'primary_class': cls,
                'license_name': lic_name,
                'flickr_url': im.get('flickr_url', ''),
                'coco_url': im.get('coco_url', ''),
                'width': im['width'],
                'height': im['height'],
            }
        )
    out.sort(key=lambda r: r['image_id'])
    return out


def select_per_class(
    val_pool: list[dict[str, Any]],
    train_pool: list[dict[str, Any]],
    classes: list[str],
    per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Val2017 first, topped up from train2017, seeded per class."""
    selected: list[dict[str, Any]] = []
    for cls in classes:
        val_cls = [r for r in val_pool if r['primary_class'] == cls]
        train_cls = [r for r in train_pool if r['primary_class'] == cls]
        chosen = seeded_sample(val_cls, per_class, seed)
        remaining = per_class - len(chosen)
        if remaining > 0:
            chosen += seeded_sample(train_cls, remaining, seed + 1)
        if len(chosen) < per_class:
            logger.warning(
                'class %r: only %d/%d eligible images available', cls, len(chosen), per_class
            )
        selected.extend(chosen)
    selected.sort(key=lambda r: r['image_id'])
    return selected


def select_side_sets(
    val_pool: list[dict[str, Any]],
    train_pool: list[dict[str, Any]],
    main_ids: set[int],
    side_sets: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Disjoint side-set samples drawn from the leftover eligible pool
    (any class), seeded, in image-id order for determinism."""
    leftover = [r for r in (*val_pool, *train_pool) if r['image_id'] not in main_ids]
    leftover.sort(key=lambda r: r['image_id'])
    out: dict[str, list[dict[str, Any]]] = {}
    used: set[int] = set()
    for offset, (name, n) in enumerate(side_sets.items()):
        pool = [r for r in leftover if r['image_id'] not in used]
        chosen = seeded_sample(pool, n, seed + 100 + offset)
        used.update(r['image_id'] for r in chosen)
        out[name] = sorted(chosen, key=lambda r: r['image_id'])
    return out


def verify_or_write_manifest(manifest_path: Path, selected: list[dict[str, Any]]) -> None:
    """Compare ``selected`` against the committed pin, or write it if this
    is the first run (bootstrap). Raises on drift."""
    pinned = load_pinned_manifest(manifest_path)
    pin_rows = [
        {
            'image_id': r['image_id'],
            'file_name': r['file_name'],
            'split': r['split'],
            'primary_class': r['primary_class'],
            'license_name': r['license_name'],
            'flickr_url': r['flickr_url'],
        }
        for r in selected
    ]
    if pinned is None:
        write_json(manifest_path, pin_rows)
        logger.info('wrote new pinned manifest %s (%d images)', manifest_path, len(pin_rows))
        return
    pinned_ids = [r['image_id'] for r in pinned]
    selected_ids = [r['image_id'] for r in pin_rows]
    if pinned_ids != selected_ids:
        raise FetchError(
            f'{manifest_path}: selection drift -- pinned {len(pinned_ids)} image ids, '
            f'recomputed {len(selected_ids)} differ. The annotation source or selection '
            'code changed since this manifest was pinned; investigate before re-pinning.'
        )


# =============================================================================
# Network-touching orchestration
# =============================================================================


def ensure_annotations(cache_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    zip_path = cache_dir / 'annotations_trainval2017.zip'
    download(ANNOTATIONS_URL, zip_path, expected_sha256=ANNOTATIONS_SHA256)
    extract_dir = cache_dir / 'annotations_trainval2017'
    val_json = extract_dir / 'annotations' / 'instances_val2017.json'
    train_json = extract_dir / 'annotations' / 'instances_train2017.json'
    if not (val_json.is_file() and train_json.is_file()):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract('annotations/instances_val2017.json', extract_dir)
            zf.extract('annotations/instances_train2017.json', extract_dir)
    return (
        json.loads(val_json.read_text(encoding='utf-8')),
        json.loads(train_json.read_text(encoding='utf-8')),
    )


def download_images(rows: list[dict[str, Any]], images_dir: Path) -> None:
    for row in rows:
        url = IMAGE_URL_TMPL[row['split']].format(file_name=row['file_name'])
        dest = images_dir / row['file_name']
        download(url, dest)
        row['sha256'] = sha256_file(dest)


def write_coco_gt(
    path: Path,
    rows: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    categories: list[dict[str, Any]],
) -> None:
    image_ids = {r['image_id'] for r in rows}
    payload = {
        'images': [
            {
                'id': r['image_id'],
                'file_name': r['file_name'],
                'width': r['width'],
                'height': r['height'],
            }
            for r in rows
        ],
        'annotations': [
            ann for img_id in image_ids for ann in annotations_by_image.get(img_id, [])
        ],
        'categories': categories,
    }
    write_json(path, payload)


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--out', required=True, type=Path, help='Output directory, e.g. data/samples/coco_va'
    )
    p.add_argument(
        '--classes', default=','.join(DEFAULT_CLASSES), help='Comma-separated COCO class names'
    )
    p.add_argument(
        '--per-class', type=int, default=80, help='Images per class (ignored if --n given)'
    )
    p.add_argument(
        '--n', type=int, default=None, help='Total image override (split evenly across classes)'
    )
    p.add_argument('--seed', type=int, default=20260925)
    p.add_argument(
        '--licenses', default=','.join(DEFAULT_LICENSES), help='Comma-separated license aliases'
    )
    p.add_argument(
        '--side-sets',
        default='',
        help="e.g. 'upload=12,post_promote=24'; disjoint samples from the leftover pool",
    )
    p.add_argument(
        '--manifest', type=Path, default=None, help='Pinned manifest path (verify or write)'
    )
    p.add_argument('--cache-dir', type=Path, default=Path('cache/datasets'))
    p.add_argument(
        '--skip-download', action='store_true', help='Compute selection/manifests only, no images'
    )
    return p


def parse_side_sets(spec: str) -> dict[str, int]:
    if not spec:
        return {}
    out: dict[str, int] = {}
    for part in spec.split(','):
        name, _, n = part.partition('=')
        if not name or not n.isdigit():
            raise FetchError(f'invalid --side-sets entry {part!r}; expected name=count')
        out[name] = int(n)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    allowed_licenses = resolve_license_names(
        [t.strip() for t in args.licenses.split(',') if t.strip()]
    )
    side_sets = parse_side_sets(args.side_sets)
    per_class = args.n // len(classes) if args.n else args.per_class

    val_data, train_data = ensure_annotations(args.cache_dir)
    category_name_to_id = {c['name']: c['id'] for c in val_data['categories']}
    category_id_to_name = {v: k for k, v in category_name_to_id.items()}
    missing = [c for c in classes if c not in category_name_to_id]
    if missing:
        raise FetchError(f'unknown COCO class name(s): {missing}')
    target_class_ids = {category_name_to_id[c] for c in classes}

    license_id_to_name = {lic['id']: lic['name'] for lic in val_data['licenses']}

    val_primary = primary_class_per_image(
        val_data['images'], val_data['annotations'], category_id_to_name, target_class_ids
    )
    train_primary = primary_class_per_image(
        train_data['images'], train_data['annotations'], category_id_to_name, target_class_ids
    )
    val_pool = eligible_images(
        val_data['images'], val_primary, license_id_to_name, allowed_licenses, 'val2017'
    )
    train_pool = eligible_images(
        train_data['images'], train_primary, license_id_to_name, allowed_licenses, 'train2017'
    )

    selected = select_per_class(val_pool, train_pool, classes, per_class, args.seed)
    if args.manifest is not None:
        verify_or_write_manifest(args.manifest, selected)

    side_selected = select_side_sets(
        val_pool, train_pool, {r['image_id'] for r in selected}, side_sets, args.seed
    )

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_images(selected, args.out / 'images')
        for name, rows in side_selected.items():
            download_images(rows, args.out / name)

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for split_data in (val_data, train_data):
        for ann in split_data['annotations']:
            if ann['category_id'] in target_class_ids and not ann.get('iscrowd', 0):
                annotations_by_image.setdefault(ann['image_id'], []).append(ann)
    categories = [{'id': category_name_to_id[c], 'name': c} for c in classes]
    write_coco_gt(args.out / 'coco_gt.json', selected, annotations_by_image, categories)

    write_csv(
        args.out / 'ATTRIBUTION.csv',
        (
            {
                'image_id': r['image_id'],
                'file_name': r['file_name'],
                'license_name': r['license_name'],
                'flickr_url': r['flickr_url'],
            }
            for r in selected
        ),
        ('image_id', 'file_name', 'license_name', 'flickr_url'),
    )
    for name, rows in side_selected.items():
        write_csv(
            args.out / name / 'ATTRIBUTION.csv',
            (
                {
                    'image_id': r['image_id'],
                    'file_name': r['file_name'],
                    'license_name': r['license_name'],
                    'flickr_url': r['flickr_url'],
                }
                for r in rows
            ),
            ('image_id', 'file_name', 'license_name', 'flickr_url'),
        )

    summary = {
        'seed': args.seed,
        'classes': classes,
        'per_class': per_class,
        'licenses': sorted(allowed_licenses),
        'n_selected': len(selected),
        'class_counts': {c: sum(1 for r in selected if r['primary_class'] == c) for c in classes},
        'side_sets': {name: len(rows) for name, rows in side_selected.items()},
    }
    write_json(args.out / 'SELECTION.json', summary)
    logger.info('done: %s', json.dumps(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except FetchError as exc:
        logger.error('%s', exc)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
