#!/usr/bin/env python3
"""Fetch a license-filtered, pinned Open Images V7 plate subset (G-05,
runbook §3.2/§3.3).

Builds the region (license-plate) sample dataset: validation+test images
carrying at least one "Vehicle registration plate" (``/m/01jfm_``) box,
filtered to images whose per-image ``License`` metadata is CC BY 2.0
(Open Images' annotations are CC BY 4.0 from Google, but the underlying
*images* are individually Flickr-sourced -- see the runbook's dataset
survey table for why Open Images was chosen over CCPD / Roboflow /
UniDataPro).

Pipeline:

1. Download the class-description CSV, the val+test bbox CSVs, and the
   val+test images-with-rotation CSVs to ``--cache-dir``.
2. Resolve the plate MID by *display name* (never hardcode ``/m/01jfm_``
   directly in the filter -- resolve it so a class-description refresh
   can't silently point at the wrong id).
3. Keep boxes with ``IsGroupOf=0`` and area >= 0.1% of the frame; an
   image is eligible if it has at least one such box AND its own
   ``License`` column is exactly the CC BY 2.0 URL.
4. Sample ``--n`` images, deterministically (``--seed``), across val+test.
5. Verify against the committed pinned manifest
   (``scripts/datasets/manifests/oi_plates_300.json``), or write it.
6. Download every selected image, write YOLO-format labels
   (``labels/<split>/<ImageID>.txt``, class 0 = ``license_plate``),
   ``data.yaml``, ``ATTRIBUTION.csv`` (ImageID, OriginalURL, Author,
   License), and ``SELECTION.json``.

Run this on the **host** (e.g. via ``make sample-plates``, or directly
with the project venv), never via ``docker compose exec`` — the same
read-only-mount caveat as ``fetch_coco_subset.py`` applies if ``--out``
resolves under ``/data/source`` inside a container.

Usage::

    python scripts/datasets/fetch_openimages_plates.py --out data/samples/oi_plates \\
        --n 300 --seed 20260925 \\
        --manifest scripts/datasets/manifests/oi_plates_300.json
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
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
logger = logging.getLogger('fetch_openimages_plates')

CLASS_DESC_URL = 'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv'
BBOX_URL = {
    'validation': 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',
    'test': 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv',
}
IMAGES_URL = {
    'validation': 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv',
    'test': 'https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv',
}
IMAGE_DOWNLOAD_TMPL = 'https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg'

PLATE_DISPLAY_NAME = 'Vehicle registration plate'
ALLOWED_LICENSE_URL = 'https://creativecommons.org/licenses/by/2.0/'
MIN_BOX_AREA_FRACTION = 0.001  # 0.1% of the frame, per the runbook selection rule
REGION_CLASS_NAME = 'license_plate'


# =============================================================================
# Pure parsing / selection logic (unit-testable without network)
# =============================================================================


def resolve_plate_mid(class_desc_csv_text: str) -> str:
    reader = csv.DictReader(io.StringIO(class_desc_csv_text))
    for row in reader:
        if row['DisplayName'] == PLATE_DISPLAY_NAME:
            return row['LabelName']
    raise FetchError(f'{PLATE_DISPLAY_NAME!r} not found in class-descriptions CSV')


def eligible_boxes(bbox_csv_text: str, plate_mid: str) -> dict[str, list[dict[str, float]]]:
    """``ImageID -> [box, ...]`` for non-group-of plate boxes, any area
    (the area-threshold eligibility check happens per-image in
    :func:`eligible_images`, since one qualifying box is enough to keep
    every plate box on that image for the label file)."""
    boxes: dict[str, list[dict[str, float]]] = {}
    reader = csv.DictReader(io.StringIO(bbox_csv_text))
    for row in reader:
        if row['LabelName'] != plate_mid:
            continue
        if row['IsGroupOf'] == '1':
            continue
        boxes.setdefault(row['ImageID'], []).append(
            {
                'x_min': float(row['XMin']),
                'x_max': float(row['XMax']),
                'y_min': float(row['YMin']),
                'y_max': float(row['YMax']),
            }
        )
    return boxes


def eligible_images(
    boxes_by_image: dict[str, list[dict[str, float]]],
    images_csv_text: str,
    split: str,
) -> list[dict[str, Any]]:
    """Images with >=1 qualifying box and an allowed license, sorted by
    ImageID for deterministic downstream sampling."""
    out = []
    reader = csv.DictReader(io.StringIO(images_csv_text))
    for row in reader:
        image_id = row['ImageID']
        boxes = boxes_by_image.get(image_id)
        if not boxes:
            continue
        if row['License'] != ALLOWED_LICENSE_URL:
            continue
        qualifies = any(
            (b['x_max'] - b['x_min']) * (b['y_max'] - b['y_min']) >= MIN_BOX_AREA_FRACTION
            for b in boxes
        )
        if not qualifies:
            continue
        out.append(
            {
                'image_id': image_id,
                'split': split,
                'license': row['License'],
                'author': row.get('Author', ''),
                'original_url': row.get('OriginalURL', ''),
                'boxes': boxes,
            }
        )
    out.sort(key=lambda r: r['image_id'])
    return out


def select_images(pool: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    return sorted(seeded_sample(pool, n, seed), key=lambda r: r['image_id'])


def verify_or_write_manifest(manifest_path: Path, selected: list[dict[str, Any]]) -> None:
    pinned = load_pinned_manifest(manifest_path)
    pin_rows = [
        {
            'image_id': r['image_id'],
            'split': r['split'],
            'license': r['license'],
            'author': r['author'],
            'original_url': r['original_url'],
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
            f'recomputed {len(selected_ids)} differ.'
        )


def yolo_label_lines(boxes: list[dict[str, float]]) -> list[str]:
    lines = []
    for b in boxes:
        cx = (b['x_min'] + b['x_max']) / 2
        cy = (b['y_min'] + b['y_max']) / 2
        w = b['x_max'] - b['x_min']
        h = b['y_max'] - b['y_min']
        lines.append(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    return lines


# =============================================================================
# Network-touching orchestration
# =============================================================================


def fetch_csv_text(url: str, cache_dir: Path, name: str) -> str:
    dest = cache_dir / name
    download(url, dest)
    return dest.read_text(encoding='utf-8')


def download_images(rows: list[dict[str, Any]], images_dir: Path) -> None:
    for row in rows:
        url = IMAGE_DOWNLOAD_TMPL.format(split=row['split'], image_id=row['image_id'])
        dest = images_dir / f'{row["image_id"]}.jpg'
        download(url, dest)
        row['sha256'] = sha256_file(dest)


def write_labels_and_data_yaml(out: Path, selected: list[dict[str, Any]]) -> None:
    for row in selected:
        label_dir = out / 'labels' / row['split']
        label_dir.mkdir(parents=True, exist_ok=True)
        (label_dir / f'{row["image_id"]}.txt').write_text(
            '\n'.join(yolo_label_lines(row['boxes'])) + '\n', encoding='utf-8'
        )
    # F-76: train and val must point at different split directories -- a
    # data.yaml that lists the same directory under both (as this used to,
    # both images/validation) makes YOLO-dataset consumers like
    # eval_regions_vs_gt.py double-count and double-score every image in
    # that overlap. This is a 2-way sample (Open Images val+test only, no
    # train pool of its own), so train takes the validation-sourced split
    # and val takes the test-sourced split -- both real, disjoint splits.
    (out / 'data.yaml').write_text(
        f'path: {out}\ntrain: images/validation\nval: images/test\n'
        f'names:\n  0: {REGION_CLASS_NAME}\n',
        encoding='utf-8',
    )


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--out', required=True, type=Path, help='Output directory, e.g. data/samples/oi_plates'
    )
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--seed', type=int, default=20260925)
    p.add_argument('--manifest', type=Path, default=None)
    p.add_argument('--cache-dir', type=Path, default=Path('cache/datasets'))
    p.add_argument('--skip-download', action='store_true')
    return p


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    class_desc = fetch_csv_text(CLASS_DESC_URL, args.cache_dir, 'oi_class_descriptions.csv')
    plate_mid = resolve_plate_mid(class_desc)

    pool: list[dict[str, Any]] = []
    for split in ('validation', 'test'):
        bbox_text = fetch_csv_text(BBOX_URL[split], args.cache_dir, f'oi_{split}_bbox.csv')
        boxes_by_image = eligible_boxes(bbox_text, plate_mid)
        images_text = fetch_csv_text(IMAGES_URL[split], args.cache_dir, f'oi_{split}_images.csv')
        pool.extend(eligible_images(boxes_by_image, images_text, split))
    pool.sort(key=lambda r: r['image_id'])

    selected = select_images(pool, args.n, args.seed)
    if args.manifest is not None:
        verify_or_write_manifest(args.manifest, selected)

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        for split in ('validation', 'test'):
            rows = [r for r in selected if r['split'] == split]
            download_images(rows, args.out / 'images' / split)
    write_labels_and_data_yaml(args.out, selected)
    write_csv(
        args.out / 'ATTRIBUTION.csv',
        (
            {
                'image_id': r['image_id'],
                'original_url': r['original_url'],
                'author': r['author'],
                'license': r['license'],
            }
            for r in selected
        ),
        ('image_id', 'original_url', 'author', 'license'),
    )

    summary = {
        'seed': args.seed,
        'n_selected': len(selected),
        'split_counts': {
            split: sum(1 for r in selected if r['split'] == split)
            for split in ('validation', 'test')
        },
    }
    write_json(args.out / 'SELECTION.json', summary)
    logger.info('done: %s', summary)
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
