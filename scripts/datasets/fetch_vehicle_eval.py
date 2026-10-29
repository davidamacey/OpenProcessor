#!/usr/bin/env python3
"""Fetch the vehicle-heavy Open Images validation set of experiment Z2 (public data only).

Selects ~8,000 images from Open Images v7 validation (CC BY 2.0, seed 20261104, see
``vehicle_eval_selection.py``), disjoint from every earlier pin given with ``--exclude-manifest``,
downloads them, and writes next to ``images/``: ``coco_gt.json`` (the dataset's own boxes in COCO
form), ``ATTRIBUTION.csv``, the pin (``--manifest``) and ``disjointness.json``.

    python scripts/datasets/fetch_vehicle_eval.py --out <dir> --cache-dir <cache> \\
        --manifest scripts/datasets/manifests/vehicle_eval_8000.json \\
        --exclude-manifest scripts/datasets/manifests/cluster_eval_3000.json \\
        --exclude-manifest scripts/datasets/manifests/cluster_eval_fresh.json \\
        --exclude-manifest scripts/datasets/manifests/coco_va_200.json

Disjointness is checked twice: by file name before the draw, and by SHA-256 of the bytes after the
download against every hash the excluded manifests carry (a manifest without hashes, such as the
COCO-200 list, is hashed from its COCO val2017 download into ``--cache-dir``). Any shared hash fails
the run. Run it on the host, not in a container.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.datasets import fetch_cluster_eval as fce, vehicle_eval_selection as ves
from scripts.datasets._common import FetchError, download, sha256_file, write_csv, write_json
from scripts.datasets.cluster_eval_selection import openimages_images
from scripts.datasets.coco_fetch import IMAGE_URL_TMPL
from scripts.datasets.fetch_openimages_plates import (
    BBOX_URL,
    CLASS_DESC_URL,
    IMAGES_URL,
    fetch_csv_text,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('fetch_vehicle_eval')

SPLIT = 'validation'


def excluded_hashes(paths: list[Path], cache_dir: Path) -> set[str]:
    """SHA-256 of every image the excluded manifests list.

    A pin carries the hashes of its base images and variants. A plain list of COCO val2017 rows
    (no hashes) is downloaded from the COCO site and hashed.
    """
    hashes: set[str] = set()
    for path in paths:
        doc = json.loads(path.read_text())
        if isinstance(doc, list):
            for r in doc:
                if r.get('sha256'):
                    hashes.add(r['sha256'])
                    continue
                url = IMAGE_URL_TMPL[r.get('split', 'val2017')].format(file_name=r['file_name'])
                hashes.add(sha256_file(download(url, cache_dir / 'excluded' / r['file_name'])))
            continue
        hashes |= {r['sha256'] for r in doc.get('images', []) if r.get('sha256')}
        for nd in doc.get('near_duplicates', []):
            hashes |= {v['sha256'] for v in nd['variants'] if v.get('sha256')}
    return hashes


def disjointness_report(
    base: list[dict[str, Any]], names: frozenset[str], hashes: set[str]
) -> dict[str, Any]:
    """What the new set shares with the excluded ones, by file name and by content hash."""
    mine = {r['file_name'] for r in base}
    mine_hashes = [r['sha256'] for r in base]
    shared = ves.overlap(mine, set(mine_hashes), set(names), hashes)
    return {
        'images': len(base),
        'excluded_names': len(names),
        'excluded_hashes': len(hashes),
        'shared_names': shared['names'],
        'shared_hashes': shared['hashes'],
        'duplicate_hashes_inside_the_set': len(mine_hashes) - len(set(mine_hashes)),
    }


def select(cache_dir: Path, seed: int, scale: float, exclude: frozenset[str]) -> dict[str, Any]:
    mids = fce.oi_class_ids(fetch_csv_text(CLASS_DESC_URL, cache_dir, 'oi_class_descriptions.csv'))
    boxes = fce.oi_boxes(fetch_csv_text(BBOX_URL[SPLIT], cache_dir, f'oi_{SPLIT}_bbox.csv'), mids)
    meta = fce.oi_metadata(
        fetch_csv_text(IMAGES_URL[SPLIT], cache_dir, f'oi_{SPLIT}_images.csv'), SPLIT
    )
    rows = openimages_images(boxes, meta)
    return {'base': ves.select_base(rows, seed, scale, exclude), 'boxes': boxes, 'rows': len(rows)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    names = fce.excluded_names(args.exclude_manifest)
    chosen = select(args.cache_dir, args.seed, args.scale, names)
    base = chosen['base']
    images_dir = args.out / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(fce.WORKERS) as pool:
        list(pool.map(lambda r: fce.fetch_one(r, images_dir), base))
    report = disjointness_report(
        base, names, excluded_hashes(args.exclude_manifest, args.cache_dir)
    )
    report['candidate_images_after_licence_filter'] = chosen['rows']
    report['by_group'] = {g: sum(r['group'] == g for r in base) for g in ves.GROUP_ORDER}
    write_json(args.out / 'disjointness.json', report)
    if (
        report['shared_names']
        or report['shared_hashes']
        or report['duplicate_hashes_inside_the_set']
    ):
        raise FetchError(f'not disjoint or duplicated, see {args.out / "disjointness.json"}')
    gt = fce.ground_truth(base, [], {'categories': [], 'annotations': []}, chosen['boxes'])
    write_json(args.out / 'coco_gt.json', gt)
    pin = ves.build_pin(base, args.seed, args.scale, sha256_file(args.out / 'coco_gt.json'))
    write_json(args.manifest, pin)
    write_csv(
        args.out / 'ATTRIBUTION.csv',
        base,
        ('file_name', 'source', 'image_id', 'license', 'author', 'attribution_url'),
        extrasaction='ignore',
    )
    return pin


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, default=Path('cache/datasets'))
    p.add_argument('--seed', type=int, default=ves.SEED)
    p.add_argument('--scale', type=float, default=1.0, help='multiplier on the group quotas')
    p.add_argument('--exclude-manifest', type=Path, action='append', default=[])
    args = p.parse_args(argv)
    try:
        pin = run(args)
    except FetchError as exc:
        logger.error('%s', exc)
        return 1
    logger.info('pinned %d images, manifest_sha256 %s', len(pin['images']), pin['manifest_sha256'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
