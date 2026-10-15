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

Throughput benchmark sets (``--bench-set N``) use a different draw: uniform over
every class, val2017 first, then train2017. They are pinned with per-image
SHA-256 and byte counts (``scripts/datasets/bench_set.py``)::

    python scripts/datasets/fetch_coco_subset.py --out data/samples/coco_bench_2000 \\
        --bench-set 2000 --seed 20261009 \\
        --manifest scripts/datasets/manifests/coco_bench_2000.json
    # re-check files later, offline
    python scripts/datasets/fetch_coco_subset.py --verify-only --out data/samples/coco_bench_2000 \\
        --manifest scripts/datasets/manifests/coco_bench_2000.json

Run this on the **host** (e.g. via ``make sample-coco`` /
``make sample-coco-readme``, or directly with the project venv), never
via ``docker compose exec``. ``--out`` is resolved relative to the
current working directory, and ``docker-compose.yml`` mounts
``${OP_SOURCE_ROOT_HOST:-./data/source}`` into the ``api`` /
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

    # Import fixture: val2017 only, plus 12 frames with no box of the chosen
    # classes (their other annotations stay in coco_gt.json). One pin covers
    # positives and negatives; crowd boxes stay in coco_gt.json, flagged.
    python scripts/datasets/fetch_coco_subset.py --out data/samples/coco_import \\
        --classes car,truck,bus --per-class 28 --negatives 12 --val-only --licenses by \\
        --manifest scripts/datasets/manifests/coco_import_88.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.datasets import bench_set
from scripts.datasets._common import FetchError, write_csv, write_json
from scripts.datasets.coco_fetch import (
    ANNOTATIONS_SHA256,
    IMAGE_URL_TMPL,
    download_images,
    ensure_annotations,
    write_coco_gt,
)
from scripts.datasets.coco_selection import (
    DEFAULT_CLASSES,
    DEFAULT_LICENSES,
    eligible_images,
    parse_side_sets,
    primary_class_per_image,
    resolve_license_names,
    select_negatives,
    select_per_class,
    select_side_sets,
    verify_or_write_manifest,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('fetch_coco_subset')


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
    p.add_argument(
        '--val-only',
        action='store_true',
        help='Select from val2017 only (small, fast; never tops up from train2017)',
    )
    p.add_argument(
        '--negatives',
        type=int,
        default=0,
        help='Also select N val2017 images with no box of any --classes class (full annotations kept)',
    )
    p.add_argument(
        '--bench-set',
        type=int,
        default=None,
        metavar='N',
        help=(
            'Throughput benchmark set: N images drawn uniformly over all classes (val2017 first, '
            'train2017 for the rest), written as a checksum pin (--manifest, required)'
        ),
    )
    p.add_argument(
        '--verify-only',
        action='store_true',
        help='Re-hash <out>/images against the --manifest pin; no network, no annotations',
    )
    p.add_argument('--cache-dir', type=Path, default=Path('cache/datasets'))
    p.add_argument(
        '--skip-download', action='store_true', help='Compute selection/manifests only, no images'
    )
    return p


def run_bench_set(args: argparse.Namespace) -> dict[str, Any]:
    """``--bench-set N``: select, download, checksum and pin a throughput set."""
    if args.manifest is None:
        raise FetchError('--bench-set needs --manifest (the pin to write or verify)')
    names = sorted(
        resolve_license_names([t.strip() for t in args.licenses.split(',') if t.strip()])
    )
    val_data, train_data = ensure_annotations(args.cache_dir)
    lic = {x['id']: x['name'] for x in val_data['licenses']}
    val_pool = bench_set.pool_rows(val_data['images'], lic, set(names), 'val2017')
    train_pool = (
        []
        if args.val_only
        else bench_set.pool_rows(train_data['images'], lic, set(names), 'train2017')
    )
    rows = bench_set.select_uniform(val_pool, train_pool, args.bench_set, args.seed)
    pin = bench_set.write_or_check_pin(
        args.manifest,
        rows,
        {'seed': args.seed, 'licenses': names, 'annotations_sha256': ANNOTATIONS_SHA256},
        args.out / 'images',
    )
    write_csv(
        args.out / 'ATTRIBUTION.csv',
        pin['images'],
        ('image_id', 'file_name', 'split', 'license_name', 'flickr_url'),
        extrasaction='ignore',
    )
    summary = {k: pin[k] for k in ('seed', 'count', 'licenses', 'manifest_sha256')}
    summary['splits'] = {s: sum(r['split'] == s for r in pin['images']) for s in IMAGE_URL_TMPL}
    write_json(args.out / 'SELECTION.json', summary)
    logger.info('done: %s', json.dumps(summary))
    return summary


def run_verify(args: argparse.Namespace) -> int:
    if args.manifest is None:
        raise FetchError('--verify-only needs --manifest')
    pin = bench_set.load_pin(args.manifest)
    problems = bench_set.verify_images(pin, args.out / 'images')
    for line in problems[:20]:
        logger.error('%s', line)
    logger.info('%d images checked, %d problem(s)', pin['count'], len(problems))
    return 1 if problems else 0


def _ground_truth(
    val_data: dict[str, Any],
    train_data: dict[str, Any],
    classes: list[str],
    category_name_to_id: dict[str, int],
    negatives: list[dict[str, Any]],
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Annotations per selected image and the category list of ``coco_gt.json``."""
    target_class_ids = {category_name_to_id[c] for c in classes}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for split_data in (val_data, train_data):
        for ann in split_data['annotations']:
            # Crowd boxes stay (flagged ``iscrowd``): the file is COCO-faithful, and
            # the import fixture exercises ``coco_crowd_skipped`` from them.
            if ann['category_id'] in target_class_ids:
                annotations_by_image.setdefault(ann['image_id'], []).append(ann)
    categories = [{'id': category_name_to_id[c], 'name': c} for c in classes]
    if negatives:
        # A negative keeps every annotation it has (other categories), so the
        # ground truth stays COCO-faithful; it is negative only for ``classes``.
        negative_ids = {r['image_id'] for r in negatives}
        for ann in val_data['annotations']:
            if ann['image_id'] in negative_ids:
                annotations_by_image.setdefault(ann['image_id'], []).append(ann)
        used_cat_ids = {
            a['category_id'] for i in negative_ids for a in annotations_by_image.get(i, [])
        }
        categories = [
            {'id': c['id'], 'name': c['name']}
            for c in val_data['categories']
            if c['name'] in classes or c['id'] in used_cat_ids
        ]
    return annotations_by_image, categories


def _write_attribution(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.bench_set is not None:
        return run_bench_set(args)
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
    if args.val_only:
        train_data = {**train_data, 'images': [], 'annotations': []}
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

    side_selected = select_side_sets(
        val_pool, train_pool, {r['image_id'] for r in selected}, side_sets, args.seed
    )

    negatives: list[dict[str, Any]] = []
    if args.negatives > 0:
        negatives = select_negatives(
            val_data['images'],
            val_data['annotations'],
            target_class_ids,
            license_id_to_name,
            allowed_licenses,
            'val2017',
            args.negatives,
            args.seed,
        )
        if len(negatives) < args.negatives:
            logger.warning('only %d/%d negative images available', len(negatives), args.negatives)
    if args.manifest is not None:
        # One pin covers the positives and the negatives (``primary_class``
        # is '' for a negative) so a drifting negative draw is also caught.
        verify_or_write_manifest(
            args.manifest, sorted([*selected, *negatives], key=lambda r: r['image_id'])
        )

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_images(selected, args.out / 'images')
        download_images(negatives, args.out / 'images')
        for name, rows in side_selected.items():
            download_images(rows, args.out / name)

    annotations_by_image, categories = _ground_truth(
        val_data, train_data, classes, category_name_to_id, negatives
    )
    write_coco_gt(
        args.out / 'coco_gt.json', [*selected, *negatives], annotations_by_image, categories
    )

    _write_attribution(args.out / 'ATTRIBUTION.csv', [*selected, *negatives])
    for name, rows in side_selected.items():
        _write_attribution(args.out / name / 'ATTRIBUTION.csv', rows)

    summary = {
        'seed': args.seed,
        'classes': classes,
        'per_class': per_class,
        'licenses': sorted(allowed_licenses),
        'n_selected': len(selected),
        'class_counts': {c: sum(1 for r in selected if r['primary_class'] == c) for c in classes},
        'side_sets': {name: len(rows) for name, rows in side_selected.items()},
        'n_negatives': len(negatives),
        'negative_image_ids': [r['image_id'] for r in negatives],
        'val_only': bool(args.val_only),
    }
    write_json(args.out / 'SELECTION.json', summary)
    logger.info('done: %s', json.dumps(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_only:
            return run_verify(args)
        run(args)
    except FetchError as exc:
        logger.error('%s', exc)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
