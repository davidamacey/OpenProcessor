"""Open Images V7 subsets for the face and vehicle proxies (selection logic and download).

Images in the V7 validation and test splits that carry the CC BY 2.0 licence in the image
list are eligible. A box counts when it is not a group, not a depiction, and covers at least
``min_area`` of the image. Faces: images with a qualifying "Human face" box. Vehicles: the
dominant (largest) qualifying box decides the class. Selection is seeded and sorted so a
re-run picks the same images.
"""

from __future__ import annotations

import csv
import io
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from scripts.datasets._common import FetchError, download, seeded_sample, sha256_file, write_csv
from scripts.datasets.proxy_manifest import (
    Licence,
    build_manifest,
    file_entry,
    label_row,
    write_dataset as write_manifest_dir,
)


if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger('proxy_openimages')

CLASSES_URL = 'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv'
BBOX_URL = 'https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv'
IMAGES_URL = (
    'https://storage.googleapis.com/openimages/2018_04/{split}/{split}-images-with-rotation.csv'
)
IMAGE_URL = 'https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg'
SPLITS = ('validation', 'test')
LICENCE_URL = 'https://creativecommons.org/licenses/by/2.0/'
ANNOTATION_LICENCE = 'CC BY 4.0'
FACE_CLASS = 'Human face'
VEHICLE_CLASSES = ('Car', 'Truck', 'Bus', 'Motorcycle', 'Van', 'Ambulance', 'Taxi')
DOWNLOAD_THREADS = 4


def class_ids(classes_csv_text: str, names: tuple[str, ...]) -> dict[str, str]:
    """Machine id -> display name for exactly ``names``; a name that is absent is an error."""
    by_name = {row[1]: row[0] for row in csv.reader(io.StringIO(classes_csv_text)) if len(row) > 1}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise FetchError(f'Open Images class names not found in the boxable list: {missing}')
    return {by_name[n]: n for n in names}


def eligible_image_ids(images_csv_text: str) -> dict[str, dict[str, str]]:
    """ImageID -> row for images that carry the allowed licence."""
    return {
        r['ImageID']: r
        for r in csv.DictReader(io.StringIO(images_csv_text))
        if r['License'] == LICENCE_URL
    }


def qualifying_boxes(
    bbox_csv_text: str, wanted: dict[str, str], min_area: float
) -> dict[str, list[dict[str, Any]]]:
    """ImageID -> its qualifying boxes of the wanted classes (largest first)."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in csv.DictReader(io.StringIO(bbox_csv_text)):
        name = wanted.get(r['LabelName'])
        if name is None or r['IsGroupOf'] == '1' or r['IsDepiction'] == '1':
            continue
        x1, x2, y1, y2 = (float(r[k]) for k in ('XMin', 'XMax', 'YMin', 'YMax'))
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue
        out[r['ImageID']].append(
            {'class_name': name, 'box': [x1, y1, x2, y2], 'area': round(area, 6)}
        )
    for boxes in out.values():
        boxes.sort(key=lambda b: (-b['area'], b['class_name']))
    return dict(out)


def select_images(
    boxes: dict[str, list[dict[str, Any]]],
    eligible: dict[str, dict[str, str]],
    per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Up to ``per_class`` images per dominant class, seeded, sorted by ImageID."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for image_id in sorted(boxes):
        if image_id in eligible:
            by_class[boxes[image_id][0]['class_name']].append(image_id)
    chosen: list[dict[str, Any]] = []
    for name in sorted(by_class):
        for image_id in sorted(seeded_sample(by_class[name], per_class, seed)):
            row = eligible[image_id]
            chosen.append(
                {
                    'image_id': image_id,
                    'class_name': name,
                    'boxes': boxes[image_id],
                    'author': row.get('Author', ''),
                    'original_url': row.get('OriginalURL', ''),
                    'license': row['License'],
                }
            )
    return sorted(chosen, key=lambda r: r['image_id'])


def summarize(selection: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'images': len(selection),
        'per_class': dict(Counter(r['class_name'] for r in selection)),
        'per_licence': dict(Counter(r['license'] for r in selection)),
    }


def _fetch_one(row: dict[str, Any], split: str, images_dir: Path) -> dict[str, Any] | None:
    dest = images_dir / f'{row["image_id"]}.jpg'
    try:
        download(IMAGE_URL.format(split=split, image_id=row['image_id']), dest, retries=2)
    except FetchError:
        return None
    return {**row, 'bytes': dest.stat().st_size, 'sha256': sha256_file(dest)}


def fetch_images(
    selection: list[dict[str, Any]], split: str, images_dir: Path
) -> list[dict[str, Any]]:
    """Download the selection; images that are gone from the bucket are dropped and counted."""
    images_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        results = list(pool.map(lambda r: _fetch_one(r, split, images_dir), selection))
    got = [r for r in results if r is not None]
    if len(got) < len(selection):
        logger.warning(
            '%d of %d images were unavailable', len(selection) - len(got), len(selection)
        )
    return got


def write_dataset(
    out: Path,
    name: str,
    licence: Licence,
    got: list[dict[str, Any]],
    split: str,
    taxonomy_prefix: list[str],
    pin: Path | None,
) -> None:
    """Write manifest, labels, ATTRIBUTION.csv (and the optional pin) for downloaded images."""
    files = [
        file_entry(
            out,
            f'images/{r["image_id"]}.jpg',
            image_id=r['image_id'],
            class_name=r['class_name'],
            license=r['license'],
            author=r['author'],
            original_url=r['original_url'],
            boxes=r['boxes'],
        )
        for r in got
    ]
    manifest = build_manifest(name, BBOX_URL.format(split=split), licence, files, split=split)
    labels = [
        label_row(
            f['path'],
            f['class_name'],
            [*taxonomy_prefix, f['class_name']],
            boxes=f['boxes'],
        )
        for f in files
    ]
    write_manifest_dir(out, manifest, labels, pin)
    write_csv(
        out / 'ATTRIBUTION.csv',
        [
            {k: f[k] for k in ('image_id', 'author', 'license', 'original_url', 'sha256')}
            for f in files
        ],
        ['image_id', 'author', 'license', 'original_url', 'sha256'],
    )
