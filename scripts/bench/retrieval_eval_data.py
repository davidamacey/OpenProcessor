"""Public data for the retrieval baseline: a seeded COCO val2017 sample with its human captions
and instance-category sets, plus deterministic near-duplicate variants.

Sources (official COCO downloads): ``annotations_trainval2017.zip`` (captions CC BY 4.0,
SHA-256 pinned in ``scripts/datasets/coco_fetch.py``) and the val2017 images (Flickr terms).
"""

from __future__ import annotations

import io
import json
import random
import zipfile
from typing import TYPE_CHECKING, Any

from scripts.datasets._common import download, sha256_file
from scripts.datasets.coco_fetch import ANNOTATIONS_SHA256, ANNOTATIONS_URL, IMAGE_URL_TMPL


if TYPE_CHECKING:
    from pathlib import Path

SEED = 20261028
CAPTIONS_PER_IMAGE = 5


def load_annotations(cache_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (captions_val2017, instances_val2017) from the pinned archive."""
    zip_path = cache_dir / 'annotations_trainval2017.zip'
    download(ANNOTATIONS_URL, zip_path, expected_sha256=ANNOTATIONS_SHA256)
    with zipfile.ZipFile(zip_path) as zf:
        caps = json.loads(zf.read('annotations/captions_val2017.json'))
        inst = json.loads(zf.read('annotations/instances_val2017.json'))
    return caps, inst


def build_sample(
    caps: dict[str, Any], inst: dict[str, Any], n: int, seed: int
) -> list[dict[str, Any]]:
    """Seeded random n images that each have at least 5 captions; rows carry captions + categories."""
    by_img: dict[int, list[dict[str, Any]]] = {}
    for a in caps['annotations']:
        by_img.setdefault(a['image_id'], []).append(a)
    cats: dict[int, set[int]] = {}
    for a in inst['annotations']:
        cats.setdefault(a['image_id'], set()).add(a['category_id'])
    names = {c['id']: c['name'] for c in inst['categories']}
    files = {i['id']: i['file_name'] for i in caps['images']}
    eligible = sorted(i for i, anns in by_img.items() if len(anns) >= CAPTIONS_PER_IMAGE)
    picked = sorted(random.Random(seed).sample(eligible, n))
    rows = []
    for iid in picked:
        anns = sorted(by_img[iid], key=lambda a: a['id'])[:CAPTIONS_PER_IMAGE]
        rows.append(
            {
                'coco_id': iid,
                'file_name': files[iid],
                'captions': [' '.join(a['caption'].split()) for a in anns],
                'categories': sorted(names[c] for c in cats.get(iid, set())),
            }
        )
    return rows


def download_sample(rows: list[dict[str, Any]], images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        dest = images_dir / row['file_name']
        download(IMAGE_URL_TMPL['val2017'].format(file_name=row['file_name']), dest)
        row['sha256'] = sha256_file(dest)


# Variants are applied to the original JPEG and saved as JPEG; the name encodes the transform.
VARIANTS = ('recompress_q60', 'resize_50', 'crop_90', 'brighten_1.15')


def make_variant(src: Path, kind: str) -> bytes:
    from PIL import Image, ImageEnhance

    with Image.open(src) as opened:
        im = opened.convert('RGB')
    w, h = im.size
    quality = 90
    if kind == 'recompress_q60':
        quality = 60
    elif kind == 'resize_50':
        im = im.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.LANCZOS)
    elif kind == 'crop_90':
        dx, dy = int(w * 0.05), int(h * 0.05)
        im = im.crop((dx, dy, w - dx, h - dy))
    elif kind == 'brighten_1.15':
        im = ImageEnhance.Brightness(im).enhance(1.15)
    else:
        raise ValueError(kind)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=quality)
    return buf.getvalue()
