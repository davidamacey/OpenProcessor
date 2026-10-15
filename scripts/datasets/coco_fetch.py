"""Network-touching COCO steps: the annotations archive (pinned by SHA-256), the
image downloads, and the COCO-format ground-truth subset written next to them."""

from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING, Any

from scripts.datasets import bench_set
from scripts.datasets._common import download, require_basename, sha256_file, write_json


if TYPE_CHECKING:
    from pathlib import Path

ANNOTATIONS_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
# Verified by direct download 2026-09-25 (runbook §3.1: 252,907,541 bytes).
# COCO's file at this URL has been stable since the 2017 release.
ANNOTATIONS_SHA256 = '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'

IMAGE_URL_TMPL = bench_set.IMAGE_URL_TMPL


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
        dest = images_dir / require_basename(row['file_name'])
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
