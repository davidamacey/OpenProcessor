"""Seeded, checksum-pinned COCO 2017 benchmark sets (all classes, license-filtered).

The class-balanced selection in ``fetch_coco_subset.py`` is built for
curation demos. A throughput baseline wants the opposite: a uniform random
draw over every image with an allowed license, so the item count per image
looks like real photos. This module holds that selection and its pin format;
``fetch_coco_subset.py --bench-set N`` drives it.

A pin is a JSON document with the seed, the allowed licenses, the SHA-256 of
the annotation archive it was drawn from, and one row per image (id, file
name, split, license, source URL, width, height, bytes, sha256).
``manifest_hash`` is the SHA-256 of the canonical JSON of the rows, so one
short string identifies the exact bytes of a set. ``verify_images`` re-hashes
a downloaded directory against the pin and needs no network.

Sizes: val2017 holds 5,000 images, of which roughly half carry a license in
the allowed set, so 2,000 fit in val2017. Larger sets (10k, 50k, 100k) top up
from train2017 (118k images) with a separate seeded draw.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from scripts.datasets._common import (
    FetchError,
    download,
    require_basename,
    seeded_sample,
    sha256_file,
    write_json,
)


if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = 1
IMAGE_URL_TMPL = {
    'val2017': 'http://images.cocodataset.org/val2017/{file_name}',
    'train2017': 'http://images.cocodataset.org/train2017/{file_name}',
}
ROW_KEYS = (
    'image_id',
    'file_name',
    'split',
    'license_name',
    'flickr_url',
    'width',
    'height',
    'bytes',
    'sha256',
)


def pool_rows(
    images: list[dict[str, Any]],
    license_id_to_name: dict[int, str],
    allowed_license_names: set[str],
    split: str,
) -> list[dict[str, Any]]:
    """Every image with an allowed license, as pin rows, sorted by image id."""
    rows = []
    for im in images:
        name = license_id_to_name.get(im['license'])
        if name not in allowed_license_names:
            continue
        rows.append(
            {
                'image_id': im['id'],
                'file_name': im['file_name'],
                'split': split,
                'license_name': name,
                'flickr_url': im.get('flickr_url', ''),
                'width': im['width'],
                'height': im['height'],
            }
        )
    rows.sort(key=lambda r: r['image_id'])
    return rows


def select_uniform(
    val_pool: list[dict[str, Any]], train_pool: list[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    """``n`` rows: a seeded draw from val2017, topped up from train2017 (seed + 1).

    Raises when both pools together hold fewer than ``n`` rows, so a pin never
    silently shrinks. Result is sorted by image id.
    """
    if len(val_pool) + len(train_pool) < n:
        raise FetchError(
            f'only {len(val_pool) + len(train_pool)} eligible images for a set of {n}; '
            'widen --licenses or drop --val-only'
        )
    chosen = seeded_sample(val_pool, n, seed)
    if len(chosen) < n:
        chosen += seeded_sample(train_pool, n - len(chosen), seed + 1)
    return sorted(chosen, key=lambda r: r['image_id'])


def manifest_hash(rows: list[dict[str, Any]]) -> str:
    """SHA-256 of the canonical JSON of the rows (key order and spacing fixed)."""
    body = json.dumps(
        [{k: r[k] for k in ROW_KEYS} for r in rows], sort_keys=True, separators=(',', ':')
    )
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def build_pin(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    licenses: list[str],
    annotations_sha256: str,
) -> dict[str, Any]:
    return {
        'schema': SCHEMA,
        'seed': seed,
        'count': len(rows),
        'licenses': sorted(licenses),
        'annotations_sha256': annotations_sha256,
        'manifest_sha256': manifest_hash(rows),
        'images': rows,
    }


def load_pin(path: Path) -> dict[str, Any]:
    pin = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(pin, dict) or pin.get('schema') != SCHEMA:
        raise FetchError(f'{path}: not a schema {SCHEMA} benchmark pin')
    rows = pin.get('images')
    if not isinstance(rows, list) or any(set(ROW_KEYS) - set(r) for r in rows):
        raise FetchError(f'{path}: image rows are missing required keys {ROW_KEYS}')
    if pin.get('count') != len(rows) or pin.get('manifest_sha256') != manifest_hash(rows):
        raise FetchError(f'{path}: count or manifest_sha256 does not match the rows')
    return pin


def verify_images(pin: dict[str, Any], images_dir: Path) -> list[str]:
    """Problems found re-hashing ``images_dir`` against the pin (empty list = clean)."""
    problems = []
    for row in pin['images']:
        path = images_dir / require_basename(row['file_name'])
        if not path.is_file():
            problems.append(f'{row["file_name"]}: missing')
        elif path.stat().st_size != row['bytes']:
            problems.append(
                f'{row["file_name"]}: {path.stat().st_size} bytes, pinned {row["bytes"]}'
            )
        elif sha256_file(path) != row['sha256']:
            problems.append(f'{row["file_name"]}: sha256 differs from the pin')
    return problems


def fetch_images(rows: list[dict[str, Any]], images_dir: Path, *, pinned: bool) -> None:
    """Download every row; fill ``bytes``/``sha256`` (first run) or enforce them (pinned)."""
    for row in rows:
        dest = images_dir / require_basename(row['file_name'])
        url = IMAGE_URL_TMPL[row['split']].format(file_name=row['file_name'])
        download(url, dest, expected_sha256=row['sha256'] if pinned else None)
        row['bytes'] = dest.stat().st_size
        row['sha256'] = sha256_file(dest)


def write_or_check_pin(
    path: Path, rows: list[dict[str, Any]], pin_args: dict[str, Any], images_dir: Path
) -> dict[str, Any]:
    """First run: download, hash and write the pin. Later runs: the recomputed
    selection must equal the pin's image ids, and every file must match its pin."""
    if not path.is_file():
        fetch_images(rows, images_dir, pinned=False)
        pin = build_pin(rows, **pin_args)
        write_json(path, pin)
        return pin
    pin = load_pin(path)
    if [r['image_id'] for r in pin['images']] != [r['image_id'] for r in rows]:
        raise FetchError(
            f'{path}: selection drift: the recomputed ids differ from the pin. The annotation '
            'source, seed, license set or selection code changed; investigate before re-pinning.'
        )
    if pin['annotations_sha256'] != pin_args['annotations_sha256']:
        raise FetchError(f'{path}: the annotation archive differs from the pinned one')
    fetch_images(pin['images'], images_dir, pinned=True)
    return pin
