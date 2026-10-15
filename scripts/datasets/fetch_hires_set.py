#!/usr/bin/env python3
"""Fetch a license-filtered, pinned set of high-resolution public photos ("set B").

Throughput work needs images where JPEG decode and resize cost something: COCO frames are
640 x 480, a modern camera photo is 12 to 24 MP. Wikimedia Commons lists the original file of
every image with its size, SHA-1 and license, and its featured-picture category is full of
camera-original JPEGs. The set is drawn from that category: JPEG only, between ``--min-pixels``
and ``--max-pixels``, license one of CC0, public domain, CC BY, CC BY-SA (the short name Commons
reports for the file), then a seeded sample of ``--n``.

The pin has the same shape as the COCO pin (``bench_set.load_pin`` and the baseline harness read
both): one row per image with its SHA-256, byte size, dimensions, source URL (in the field the
COCO pin calls ``flickr_url``) and license, and a ``manifest_sha256`` over the rows. The first run
lists the category, samples, downloads and writes the pin; a later run downloads exactly the
pinned files and checks every hash, so the set does not depend on the category staying the same.

    fetch_hires_set.py --out data/samples/hires_500 --n 500 --seed 20261010 \\
        --manifest scripts/datasets/manifests/commons_hires_500.json

Run on the host, not inside a container. Images are not committed; the pin and
``ATTRIBUTION.csv`` (page id, title, author, license, description page) are.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.datasets._common import (
    USER_AGENT,
    FetchError,
    download,
    require_basename,
    seeded_sample,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.datasets.bench_set import build_pin, load_pin, verify_images


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('fetch_hires_set')

API_URL = 'https://commons.wikimedia.org/w/api.php'
CATEGORY = 'Category:Featured_pictures_on_Wikimedia_Commons'
ALLOWED_LICENSE = re.compile(r'^(CC0|Public domain|PD\b|CC BY(-SA)?(\s|$))', re.IGNORECASE)
DOWNLOAD_THREADS = 4  # Commons asks for polite clients


def api_get(url: str, attempts: int = 8) -> dict[str, Any]:
    """One API request; a 429 waits as long as the server asks (or 15 s per attempt)."""
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310  URL is built from the hardcoded public API host
                result: dict[str, Any] = json.load(resp)
                return result
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts:
                raise
            time.sleep(float(exc.headers.get('Retry-After') or 15 * attempt))
    raise FetchError('unreachable')


def api_pages(category: str) -> list[dict[str, Any]]:
    """Every file in ``category`` with its image info (one API request per 50 files)."""
    params = {
        'action': 'query',
        'format': 'json',
        'generator': 'categorymembers',
        'gcmtitle': category,
        'gcmtype': 'file',
        'gcmlimit': '50',
        'prop': 'imageinfo',
        'iiprop': 'url|size|sha1|mime|extmetadata',
        'iiextmetadatafilter': 'LicenseShortName|Artist',
    }
    pages: list[dict[str, Any]] = []
    while True:
        url = f'{API_URL}?{urllib.parse.urlencode(params)}'
        doc = api_get(url)
        pages.extend(doc.get('query', {}).get('pages', {}).values())
        if 'continue' not in doc:
            return pages
        params.update(doc['continue'])
        time.sleep(1.0)


def candidate_rows(
    pages: list[dict[str, Any]], min_pixels: int, max_pixels: int
) -> list[dict[str, Any]]:
    """JPEGs in the size window with an allowed license, sorted by page id."""
    out = []
    for page in pages:
        info = (page.get('imageinfo') or [{}])[0]
        meta = info.get('extmetadata', {})
        license_name = meta.get('LicenseShortName', {}).get('value', '')
        pixels = info.get('width', 0) * info.get('height', 0)
        if info.get('mime') != 'image/jpeg' or not min_pixels <= pixels <= max_pixels:
            continue
        if not ALLOWED_LICENSE.match(license_name):
            continue
        out.append(
            {
                'page_id': page['pageid'],
                'title': page['title'].removeprefix('File:'),
                'url': info['url'].split('?', 1)[0],
                'description_url': info.get('descriptionurl', ''),
                'width': info['width'],
                'height': info['height'],
                'license': license_name,
                'author': html.unescape(
                    re.sub(r'<[^>]+>', '', meta.get('Artist', {}).get('value', ''))
                ),
            }
        )
    out.sort(key=lambda r: r['page_id'])
    return out


def pin_row(cand: dict[str, Any], size: int, digest: str) -> dict[str, Any]:
    """One pin row in the COCO pin shape (``flickr_url`` holds the original file URL)."""
    return {
        'image_id': cand['page_id'],
        'file_name': f'{cand["page_id"]}.jpg',
        'split': 'wikimedia_commons',
        'license_name': cand['license'],
        'flickr_url': cand['url'],
        'width': cand['width'],
        'height': cand['height'],
        'bytes': size,
        'sha256': digest,
    }


def fetch_candidate(cand: dict[str, Any], images_dir: Path) -> dict[str, Any]:
    dest = images_dir / require_basename(f'{cand["page_id"]}.jpg')
    download(cand['url'], dest, retries=5)
    with Image.open(dest) as im:
        if im.size != (cand['width'], cand['height']):
            raise FetchError(f'{dest.name}: decoded size {im.size} differs from the listing')
    return pin_row(cand, dest.stat().st_size, sha256_file(dest))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--seed', type=int, default=20261010)
    p.add_argument('--min-pixels', type=int, default=12_000_000)
    p.add_argument('--max-pixels', type=int, default=24_000_000)
    p.add_argument('--category', default=CATEGORY)
    return p


def run(args: argparse.Namespace) -> dict[str, Any]:
    images_dir = args.out / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest.is_file():
        pin = load_pin(args.manifest)
        for row in pin['images']:
            download(
                row['flickr_url'],
                images_dir / require_basename(row['file_name']),
                expected_sha256=row['sha256'],
                retries=5,
            )
        problems = verify_images(pin, images_dir)
        if problems:
            raise FetchError('; '.join(problems[:5]))
        return pin
    cands = candidate_rows(api_pages(args.category), args.min_pixels, args.max_pixels)
    logger.info('%d candidates in the size and license window', len(cands))
    chosen = sorted(seeded_sample(cands, args.n, args.seed), key=lambda r: r['page_id'])
    if len(chosen) < args.n:
        raise FetchError(f'only {len(chosen)} candidates for a set of {args.n}')
    with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        rows = list(pool.map(lambda c: fetch_candidate(c, images_dir), chosen))
    pin = build_pin(rows, seed=args.seed, licenses=sorted({c['license'] for c in chosen}),
                    annotations_sha256='n/a')  # fmt: skip
    pin['source'] = f'Wikimedia Commons {args.category}, original files'
    pin['pixel_window'] = [args.min_pixels, args.max_pixels]
    write_json(args.manifest, pin)
    write_csv(
        args.out / 'ATTRIBUTION.csv',
        (
            {
                'page_id': c['page_id'],
                'title': c['title'],
                'author': c['author'],
                'license': c['license'],
                'description_url': c['description_url'],
            }
            for c in chosen
        ),
        ('page_id', 'title', 'author', 'license', 'description_url'),
    )
    return pin


def main(argv: list[str] | None = None) -> int:
    try:
        pin = run(build_parser().parse_args(argv))
    except FetchError as exc:
        logger.error('%s', exc)
        return 1
    logger.info('pinned %d images, manifest sha256 %s', pin['count'], pin['manifest_sha256'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
