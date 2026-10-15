"""Seeded, checksum-pinned Open Images v7 train set for throughput benchmarks.

Every Open Images train image carries CC BY 2.0. Candidates are the train image list in a
seeded order; the first N that download are kept (a few S3 objects are missing), hashed and
pinned (id, licence, author, source URL, bytes, width, height, SHA-256, plus a manifest
hash). ``ATTRIBUTION.csv`` is written next to ``images/``. Run on the host::

    python scripts/datasets/fetch_openimages_bench.py --out <dir> --n 100000 --seed 20261009 \\
        --cache-dir <cache> --manifest <pin.json>
    python scripts/datasets/fetch_openimages_bench.py --verify-only --out <dir> --manifest <pin.json>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.datasets._common import FetchError, download, sha256_file, write_csv, write_json


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('fetch_openimages_bench')

SCHEMA = 'openimages-bench-pin/1'
IMAGES_URL = (
    'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
)
IMAGE_URL_TMPL = 'https://s3.amazonaws.com/open-images-dataset/train/{image_id}.jpg'
ALLOWED_LICENSE_URL = 'https://creativecommons.org/licenses/by/2.0/'
ROW_KEYS = ('image_id', 'license', 'author', 'original_url', 'bytes', 'width', 'height', 'sha256')
CHUNK = 2000


def eligible_rows(images_csv_text: str) -> list[dict[str, Any]]:
    """Train images with the allowed licence, sorted by ImageID."""
    out = [
        {
            'image_id': r['ImageID'],
            'license': r['License'],
            'author': r.get('Author', ''),
            'original_url': r.get('OriginalURL', ''),
        }
        for r in csv.DictReader(io.StringIO(images_csv_text))
        if r['License'] == ALLOWED_LICENSE_URL
    ]
    out.sort(key=lambda r: r['image_id'])
    return out


def candidate_order(pool: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """The whole pool in a seeded order; the first N that fetch form the set."""
    order = list(pool)
    random.Random(seed).shuffle(order)
    return order


def manifest_hash(rows: list[dict[str, Any]]) -> str:
    body = json.dumps(
        [{k: r[k] for k in ROW_KEYS} for r in rows], sort_keys=True, separators=(',', ':')
    )
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def build_pin(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: r['image_id'])
    return {
        'schema': SCHEMA,
        'seed': seed,
        'count': len(rows),
        'license': ALLOWED_LICENSE_URL,
        'manifest_sha256': manifest_hash(rows),
        'images': [{k: r[k] for k in ROW_KEYS} for r in rows],
    }


def load_pin(path: Path) -> dict[str, Any]:
    pin: dict[str, Any] = json.loads(path.read_text(encoding='utf-8'))
    rows = pin.get('images') if isinstance(pin, dict) else None
    if pin.get('schema') != SCHEMA or not isinstance(rows, list):
        raise FetchError(f'{path}: not a {SCHEMA} pin')
    if any(set(ROW_KEYS) - set(r) for r in rows):
        raise FetchError(f'{path}: rows are missing required keys {ROW_KEYS}')
    if pin.get('count') != len(rows) or pin.get('manifest_sha256') != manifest_hash(rows):
        raise FetchError(f'{path}: count or manifest_sha256 does not match the rows')
    return pin


def verify_images(pin: dict[str, Any], images_dir: Path) -> list[str]:
    problems = []
    for row in pin['images']:
        name = f'{row["image_id"]}.jpg'
        path = images_dir / name
        if not path.is_file():
            problems.append(f'{name}: missing')
        elif path.stat().st_size != row['bytes']:
            problems.append(f'{name}: {path.stat().st_size} bytes, pinned {row["bytes"]}')
        elif sha256_file(path) != row['sha256']:
            problems.append(f'{name}: sha256 differs from the pin')
    return problems


def _fetch_one(row: dict[str, Any], images_dir: Path) -> dict[str, Any] | None:
    from PIL import Image

    dest = images_dir / f'{row["image_id"]}.jpg'
    try:
        download(IMAGE_URL_TMPL.format(image_id=row['image_id']), dest, retries=2)
        with Image.open(dest) as im:
            width, height = im.size
    except (FetchError, OSError):
        dest.unlink(missing_ok=True)
        return None
    return {
        **row,
        'bytes': dest.stat().st_size,
        'width': width,
        'height': height,
        'sha256': sha256_file(dest),
    }


def fetch_set(
    pool: list[dict[str, Any]], n: int, seed: int, images_dir: Path, workers: int
) -> list[dict[str, Any]]:
    got: list[dict[str, Any]] = []
    order = candidate_order(pool, seed)
    missing = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(order), CHUNK):
            for res in ex.map(lambda r: _fetch_one(r, images_dir), order[i : i + CHUNK]):
                if res is None:
                    missing += 1
                elif len(got) < n:
                    got.append(res)
            logger.info('%d/%d fetched, %d unavailable', len(got), n, missing)
            if len(got) >= n:
                break
    if len(got) < n:
        raise FetchError(f'only {len(got)} of {n} images could be fetched')
    # Candidates past n in the final chunk were fetched but are not in the set.
    keep = {r['image_id'] for r in got}
    for extra in {r['image_id'] for r in order[: i + CHUNK]} - keep:
        (images_dir / f'{extra}.jpg').unlink(missing_ok=True)
    return got


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--n', type=int, default=100_000)
    p.add_argument('--seed', type=int, default=20261009)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--cache-dir', type=Path, default=Path('cache/datasets'))
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args(argv)
    images_dir = args.out / 'images'
    try:
        if args.verify_only:
            pin = load_pin(args.manifest)
            problems = verify_images(pin, images_dir)
            for line in problems[:20]:
                logger.error(line)
            logger.info('verified %d images, %d problems', pin['count'], len(problems))
            return 1 if problems else 0
        if args.manifest.is_file():
            raise FetchError(f'{args.manifest} exists; delete it to re-pin or use --verify-only')
        dest = args.cache_dir / 'oi_train_images.csv'
        download(IMAGES_URL, dest)
        pool = eligible_rows(dest.read_text(encoding='utf-8'))
        rows = fetch_set(pool, args.n, args.seed, images_dir, args.workers)
        pin = build_pin(rows, args.seed)
        write_json(args.manifest, pin)
        write_csv(
            args.out / 'ATTRIBUTION.csv',
            pin['images'],
            ['image_id', 'author', 'license', 'original_url', 'sha256'],
            extrasaction='ignore',
        )
    except FetchError as exc:
        logger.error('%s', exc)
        return 1
    logger.info('pinned %d images -> %s', pin['count'], args.manifest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
