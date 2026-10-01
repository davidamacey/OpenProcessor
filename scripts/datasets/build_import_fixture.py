#!/usr/bin/env python3
"""Build the dataset-import test fixture from a fetched COCO subset (W10.15).

Input: the directory ``fetch_coco_subset.py`` writes (``coco_gt.json``,
``images/``, ``SELECTION.json``, ``ATTRIBUTION.csv``), e.g.::

    python scripts/datasets/fetch_coco_subset.py --out data/samples/coco_import \\
        --classes car,truck,bus --per-class 28 --negatives 12 --val-only \\
        --licenses by --seed 20260925 \\
        --manifest scripts/datasets/manifests/coco_import_96.json
    python scripts/datasets/build_import_fixture.py \\
        --src data/samples/coco_import --out data/samples/import_fixture

Output: four dataset layouts plus ``FIXTURE.json`` (the expected counts and
issues of each, the only place those numbers live)::

    yolo/              names {0: truck, 1: Car, 2: automobile, 3: bus}: NOT
                       registry order, ``Car`` differs by case, ``automobile``
                       is a synonym (car boxes on odd COCO image ids). Six
                       problems are injected once each.
    coco/              official boxes + category names, crowd boxes kept.
    yolo_region/       SYNTHETIC wheel boxes derived from COCO car boxes.
    yolo_region_only/  the same without the car rows.

The wheel boxes are synthetic geometry, not wheel annotations: they exercise
parent attachment, not model quality.

:func:`build_variants` is pure (no network, no COCO pixels): it takes the COCO
ground-truth dict and an ``image_writer`` callback, so tests build the same
layouts offline from faked annotations and synthetic JPEGs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TARGETS = ('car', 'truck', 'bus')
SPLITS = ('train', 'val', 'test')
# 70/15/15 by sha1 of the COCO image id.
SPLIT_CUTS = (70, 85)
YOLO_NAMES = {0: 'truck', 1: 'Car', 2: 'automobile', 3: 'bus'}
REGION_NAMES = {0: 'car', 1: 'wheel'}
MIN_BOX_PX = 2.0
MIN_CAR_HEIGHT_PX = 48.0
WHEEL_W, WHEEL_H, WHEEL_INSET = 0.28, 0.30, 0.06
# Where a standalone wheel may sit (normalized x, y, w, h), tried in order
# until one avoids every car box on the image.
STANDALONE_SLOTS = (
    (0.02, 0.02, 0.12, 0.12),
    (0.86, 0.02, 0.12, 0.12),
    (0.02, 0.86, 0.12, 0.12),
    (0.86, 0.86, 0.12, 0.12),
)


def split_of(image_id: int) -> str:
    bucket = int(hashlib.sha1(str(image_id).encode()).hexdigest(), 16) % 100
    return (
        SPLITS[0] if bucket < SPLIT_CUTS[0] else SPLITS[1] if bucket < SPLIT_CUTS[1] else SPLITS[2]
    )


def _order_key(image_id: int) -> str:
    return hashlib.sha1(f'inject:{image_id}'.encode()).hexdigest()


def _row(cls: int, x: float, y: float, w: float, h: float) -> str:
    return f'{cls} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}'


def _norm_box(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x0 = max(0.0, bbox[0])
    y0 = max(0.0, bbox[1])
    x1 = min(float(width), bbox[0] + bbox[2])
    y1 = min(float(height), bbox[1] + bbox[3])
    return (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height)


def _intersects(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _usable(ann: dict[str, Any]) -> bool:
    return not ann.get('iscrowd') and ann['bbox'][2] >= MIN_BOX_PX and ann['bbox'][3] >= MIN_BOX_PX


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def _yaml(names: dict[int, str]) -> str:
    keys = ''.join(f'{s}: images/{s}\n' for s in SPLITS)
    return keys + 'names:\n' + ''.join(f'  {i}: {n}\n' for i, n in names.items())


def _tally(layout: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``layout``: split -> {'states': Counter, 'boxes': Counter}."""
    return {
        'images_per_split': {s: sum(layout[s]['states'].values()) for s in SPLITS},
        'label_states': {
            state: sum(layout[s]['states'][state] for s in SPLITS)
            for state in ('labeled', 'negative', 'unlabeled')
        },
        'label_states_per_split': {s: dict(layout[s]['states']) for s in SPLITS},
        'boxes_per_class': dict(sum((layout[s]['boxes'] for s in SPLITS), Counter())),
    }


def _empty_layout() -> dict[str, dict[str, Any]]:
    return {s: {'states': Counter(), 'boxes': Counter()} for s in SPLITS}


def build_variants(
    annotations: dict[str, Any],
    images_meta: dict[str, Any],
    out: Path,
    image_writer: Callable[[str, Path, int, int], None],
) -> dict[str, Any]:
    """Write the four variants under ``out`` and return the ``FIXTURE.json`` dict.

    ``annotations`` is a COCO dict (``images`` / ``annotations`` /
    ``categories``). ``images_meta`` carries ``negative_image_ids`` (frames
    with no box of any target class) and optionally ``seed`` / ``manifest`` /
    ``license_note``. ``image_writer(file_name, dest, width, height)`` puts
    the image at ``dest``.
    """
    cat_name = {c['id']: c['name'] for c in annotations['categories']}
    negative_ids = set(images_meta.get('negative_image_ids', []))
    images = sorted((im for im in annotations['images']), key=lambda im: im['id'])
    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in annotations['annotations']:
        if cat_name.get(ann['category_id']) in TARGETS:
            by_image.setdefault(ann['image_id'], []).append(ann)

    def usable(image_id: int) -> list[dict[str, Any]]:
        return [a for a in by_image.get(image_id, []) if _usable(a)]

    positives = [im for im in images if im['id'] not in negative_ids and usable(im['id'])]
    negatives = [im for im in images if im['id'] in negative_ids]

    fixture: dict[str, Any] = {
        'seed': images_meta.get('seed'),
        'manifest': images_meta.get('manifest'),
        'license_note': images_meta.get(
            'license_note',
            'COCO 2017 val2017 images (CC BY 2.0 via Flickr, per-image credit in '
            'ATTRIBUTION.csv); COCO annotations CC BY 4.0.',
        ),
        'variants': {},
    }

    # --- yolo ---------------------------------------------------------------
    injectable = sorted((im for im in positives), key=lambda im: _order_key(im['id']))
    roles = (
        'missing_label',
        'class_out_of_range',
        'coords_out_of_range',
        'malformed_row',
        'polygon_row',
    )
    chosen = dict(zip(roles, injectable, strict=False))
    yolo_root = out / 'yolo'
    layout = _empty_layout()
    for im in images:
        split = split_of(im['id'])
        image_writer(
            im['file_name'],
            yolo_root / f'images/{split}/{im["file_name"]}',
            im['width'],
            im['height'],
        )
        stem = Path(im['file_name']).stem
        rows: list[str] = []
        counted: list[str] = []
        for ann in usable(im['id']):
            name = cat_name[ann['category_id']]
            if name == 'car':
                cls = 2 if im['id'] % 2 else 1
            else:
                cls = 0 if name == 'truck' else 3
            x, y, w, h = _norm_box(ann['bbox'], im['width'], im['height'])
            rows.append(_row(cls, x, y, w, h))
            counted.append(YOLO_NAMES[cls])
        role = next((r for r, c in chosen.items() if c['id'] == im['id']), None)
        if role == 'missing_label':
            layout[split]['states']['unlabeled'] += 1
            continue
        if role == 'class_out_of_range':
            rows.append('7 0.500000 0.500000 0.200000 0.200000')
        elif role == 'coords_out_of_range':
            rows.append('0 1.200000 0.500000 0.200000 0.200000')
        elif role == 'malformed_row':
            rows.append('0 0.500000 0.500000 0.200000')
        elif role == 'polygon_row':
            rows.append('0 0.100000 0.100000 0.400000 0.100000 0.250000 0.400000')
            counted.append(YOLO_NAMES[0])
        _write_text(yolo_root / f'labels/{split}/{stem}.txt', ''.join(r + '\n' for r in rows))
        layout[split]['states']['labeled' if counted else 'negative'] += 1
        layout[split]['boxes'].update(counted)
    _write_text(yolo_root / 'labels/train/orphan_label.txt', '0 0.5 0.5 0.2 0.2\n')
    _write_text(yolo_root / 'data.yaml', _yaml(YOLO_NAMES))
    fixture['variants']['yolo'] = {
        'names': {str(k): v for k, v in YOLO_NAMES.items()},
        'injected': {role: im['file_name'] for role, im in chosen.items()}
        | {'orphan_label': 'labels/train/orphan_label.txt'},
        'expected_report': _tally(layout),
        'expected_issues': {
            'label_file_missing': 1 if 'missing_label' in chosen else 0,
            'label_class_out_of_range': 1 if 'class_out_of_range' in chosen else 0,
            'label_coords_out_of_range': 1 if 'coords_out_of_range' in chosen else 0,
            'label_row_malformed': 1 if 'malformed_row' in chosen else 0,
            'yolo_polygon_to_box': 1 if 'polygon_row' in chosen else 0,
            'label_file_orphan': 1,
        },
        'negatives': len(negatives),
    }

    # --- coco ---------------------------------------------------------------
    coco_root = out / 'coco'
    layout = _empty_layout()
    per_split: dict[str, dict[str, list[Any]]] = {
        s: {'images': [], 'annotations': []} for s in SPLITS
    }
    crowd = 0
    for im in images:
        split = split_of(im['id'])
        image_writer(
            im['file_name'], coco_root / f'images/{im["file_name"]}', im['width'], im['height']
        )
        per_split[split]['images'].append(
            {k: im[k] for k in ('id', 'file_name', 'width', 'height')}
        )
        kept = [
            a for a in by_image.get(im['id'], []) if a['bbox'][2] >= MIN_BOX_PX or a.get('iscrowd')
        ]
        for ann in kept:
            per_split[split]['annotations'].append(ann)
            if ann.get('iscrowd'):
                crowd += 1
        valid = [a for a in kept if _usable(a)]
        layout[split]['states']['labeled' if valid else 'negative'] += 1
        layout[split]['boxes'].update(cat_name[a['category_id']] for a in valid)
    coco_categories = [c for c in annotations['categories'] if c['name'] in TARGETS]
    for split in SPLITS:
        _write_text(
            coco_root / f'annotations/instances_{split}.json',
            json.dumps({**per_split[split], 'categories': coco_categories}),
        )
    fixture['variants']['coco'] = {
        # detect_format() sees ``images/`` first and answers yolo, so a COCO
        # import must name its format explicitly.
        'source_format': 'coco',
        'names': {str(c['id']): c['name'] for c in coco_categories},
        'expected_report': _tally(layout),
        'expected_issues': {'coco_crowd_skipped': crowd},
        'negatives': len(negatives),
    }

    # --- yolo_region / yolo_region_only ----------------------------------------
    for variant, with_cars in (('yolo_region', True), ('yolo_region_only', False)):
        root = out / variant
        layout = _empty_layout()
        attached = 0
        standalone = 0
        standalone_done = False
        for im in [*positives, *negatives]:
            cars = [a for a in usable(im['id']) if cat_name[a['category_id']] == 'car']
            if im['id'] not in negative_ids and not cars:
                continue  # truck/bus-only frames have no car parent to nest wheels in
            split = split_of(im['id'])
            image_writer(
                im['file_name'],
                root / f'images/{split}/{im["file_name"]}',
                im['width'],
                im['height'],
            )
            rows = []
            counted = []
            norm_cars = [_norm_box(a['bbox'], im['width'], im['height']) for a in cars]
            for (x, y, w, h), ann in zip(norm_cars, cars, strict=True):
                if with_cars:
                    rows.append(_row(0, x, y, w, h))
                    counted.append('car')
                if ann['bbox'][3] >= MIN_CAR_HEIGHT_PX:
                    ww, wh, inset = w * WHEEL_W, h * WHEEL_H, w * WHEEL_INSET
                    for wx in (x + inset, x + w - inset - ww):
                        rows.append(_row(1, wx, y + h - wh, ww, wh))
                        counted.append('wheel')
                        attached += 1
            if im['id'] not in negative_ids and not standalone_done:
                for slot in STANDALONE_SLOTS:
                    if not any(_intersects(slot, c) for c in norm_cars):
                        rows.append(_row(1, *slot))
                        counted.append('wheel')
                        standalone += 1
                        standalone_done = True
                        break
            stem = Path(im['file_name']).stem
            _write_text(root / f'labels/{split}/{stem}.txt', ''.join(r + '\n' for r in rows))
            layout[split]['states']['labeled' if counted else 'negative'] += 1
            layout[split]['boxes'].update(counted)
        _write_text(root / 'data.yaml', _yaml(REGION_NAMES))
        fixture['variants'][variant] = {
            'names': {str(k): v for k, v in REGION_NAMES.items()},
            'synthetic_geometry': (
                'SYNTHETIC wheel boxes derived from COCO car boxes -- NOT wheel annotations. '
                'They exercise parent attachment, not model quality.'
            ),
            'expected_report': _tally(layout),
            'expected_issues': {},
            'attachments': attached,
            'standalone': standalone,
            'negatives': len(negatives),
        }
    return fixture


def _copy_writer(src_images: Path) -> Callable[[str, Path, int, int], None]:
    def write(file_name: str, dest: Path, _w: int, _h: int) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_images / file_name, dest)

    return write


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--src', required=True, type=Path, help='fetch_coco_subset.py output dir')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)

    gt_path = args.src / 'coco_gt.json'
    sel_path = args.src / 'SELECTION.json'
    if not gt_path.is_file() or not sel_path.is_file():
        print(f'{args.src}: coco_gt.json / SELECTION.json missing; run fetch_coco_subset.py first')
        return 1
    gt = json.loads(gt_path.read_text(encoding='utf-8'))
    selection = json.loads(sel_path.read_text(encoding='utf-8'))
    if args.out.exists() and any(args.out.iterdir()):
        print(f'{args.out} is not empty; refusing to overwrite')
        return 1
    fixture = build_variants(
        gt,
        {
            'negative_image_ids': selection.get('negative_image_ids', []),
            'seed': selection.get('seed'),
            'manifest': 'scripts/datasets/manifests/coco_import_96.json',
        },
        args.out,
        _copy_writer(args.src / 'images'),
    )
    attribution = args.src / 'ATTRIBUTION.csv'
    for variant in fixture['variants']:
        if attribution.is_file():
            shutil.copyfile(attribution, args.out / variant / 'ATTRIBUTION.csv')
    _write_text(args.out / 'FIXTURE.json', json.dumps(fixture, indent=2, sort_keys=True) + '\n')
    print(f'wrote {len(fixture["variants"])} variants to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
