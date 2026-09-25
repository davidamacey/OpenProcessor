"""Tests for scripts/datasets/fetch_openimages_plates.py (G-05).

Synthetic CSV fixtures matching Open Images' real column shapes; no
network. The real fetch was run once for a small (n=10) smoke subset per
the task's acceptance criteria and its images deleted afterward.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.datasets import fetch_openimages_plates as f


if TYPE_CHECKING:
    from pathlib import Path


CLASS_DESC_CSV = 'LabelName,DisplayName\n/m/0mkg,Accordion\n/m/01jfm_,Vehicle registration plate\n'

BBOX_HEADER = 'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n'

IMAGES_HEADER = (
    'ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,'
    'Author,Title,OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation\n'
)


def _bbox_row(image_id, xmin, xmax, ymin, ymax, *, group_of='0', label='/m/01jfm_'):
    return f'{image_id},xclick,{label},1,{xmin},{xmax},{ymin},{ymax},0,0,{group_of},0,0\n'


def _images_row(image_id, license_url, author='Someone', url='http://example.invalid/o.jpg'):
    return f'{image_id},validation,{url},{url},{license_url},http://x,{author},t,1,md5,thumb,\n'


# --- MID resolution --------------------------------------------------------


def test_resolve_plate_mid_finds_by_display_name() -> None:
    assert f.resolve_plate_mid(CLASS_DESC_CSV) == '/m/01jfm_'


def test_resolve_plate_mid_raises_if_not_found() -> None:
    with pytest.raises(f.FetchError):
        f.resolve_plate_mid('LabelName,DisplayName\n/m/0mkg,Accordion\n')


# --- box eligibility (IsGroupOf filter) ------------------------------------


def test_eligible_boxes_excludes_group_of() -> None:
    csv_text = (
        BBOX_HEADER
        + _bbox_row('img1', 0, 0.1, 0, 0.1)
        + _bbox_row('img2', 0, 0.1, 0, 0.1, group_of='1')
    )
    boxes = f.eligible_boxes(csv_text, '/m/01jfm_')
    assert list(boxes.keys()) == ['img1']


def test_eligible_boxes_ignores_other_labels() -> None:
    csv_text = BBOX_HEADER + _bbox_row('img1', 0, 0.1, 0, 0.1, label='/m/0mkg')
    assert f.eligible_boxes(csv_text, '/m/01jfm_') == {}


def test_eligible_boxes_collects_multiple_boxes_per_image() -> None:
    csv_text = (
        BBOX_HEADER + _bbox_row('img1', 0, 0.1, 0, 0.1) + _bbox_row('img1', 0.5, 0.6, 0.5, 0.6)
    )
    boxes = f.eligible_boxes(csv_text, '/m/01jfm_')
    assert len(boxes['img1']) == 2


# --- image eligibility: area threshold + exact CC BY 2.0 license -----------


def test_eligible_images_requires_area_threshold() -> None:
    # box area = 0.01 * 0.01 = 0.0001 = 0.01% < 0.1% floor -- not eligible
    boxes = {'img1': [{'x_min': 0, 'x_max': 0.01, 'y_min': 0, 'y_max': 0.01}]}
    images_csv = IMAGES_HEADER + _images_row('img1', f.ALLOWED_LICENSE_URL)
    assert f.eligible_images(boxes, images_csv, 'validation') == []


def test_eligible_images_passes_with_qualifying_box() -> None:
    boxes = {'img1': [{'x_min': 0, 'x_max': 0.1, 'y_min': 0, 'y_max': 0.1}]}  # 1% area
    images_csv = IMAGES_HEADER + _images_row('img1', f.ALLOWED_LICENSE_URL)
    out = f.eligible_images(boxes, images_csv, 'validation')
    assert [r['image_id'] for r in out] == ['img1']


def test_eligible_images_rejects_wrong_license() -> None:
    boxes = {'img1': [{'x_min': 0, 'x_max': 0.1, 'y_min': 0, 'y_max': 0.1}]}
    images_csv = IMAGES_HEADER + _images_row(
        'img1', 'https://creativecommons.org/licenses/by-nc/2.0/'
    )
    assert f.eligible_images(boxes, images_csv, 'validation') == []


def test_eligible_images_sorted_by_image_id() -> None:
    boxes = {
        'b': [{'x_min': 0, 'x_max': 0.1, 'y_min': 0, 'y_max': 0.1}],
        'a': [{'x_min': 0, 'x_max': 0.1, 'y_min': 0, 'y_max': 0.1}],
    }
    images_csv = (
        IMAGES_HEADER
        + _images_row('b', f.ALLOWED_LICENSE_URL)
        + _images_row('a', f.ALLOWED_LICENSE_URL)
    )
    out = f.eligible_images(boxes, images_csv, 'validation')
    assert [r['image_id'] for r in out] == ['a', 'b']


# --- selection determinism -------------------------------------------------


def _pool(ids):
    return [
        {
            'image_id': i,
            'split': 'validation',
            'license': f.ALLOWED_LICENSE_URL,
            'author': '',
            'original_url': '',
            'boxes': [{'x_min': 0, 'x_max': 0.1, 'y_min': 0, 'y_max': 0.1}],
        }
        for i in ids
    ]


def test_select_images_is_deterministic() -> None:
    pool = _pool(range(50))
    a = f.select_images(pool, 10, seed=3)
    b = f.select_images(pool, 10, seed=3)
    assert a == b
    assert len(a) == 10


# --- YOLO label line conversion --------------------------------------------


def test_yolo_label_lines_center_and_size() -> None:
    boxes = [{'x_min': 0.2, 'x_max': 0.4, 'y_min': 0.1, 'y_max': 0.3}]
    lines = f.yolo_label_lines(boxes)
    assert lines == ['0 0.300000 0.200000 0.200000 0.200000']


# --- manifest pin ------------------------------------------------------------


def test_verify_or_write_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = tmp_path / 'pin.json'
    selected = _pool([1, 2])
    f.verify_or_write_manifest(manifest, selected)
    pinned = json.loads(manifest.read_text())
    assert [p['image_id'] for p in pinned] == [1, 2]
    f.verify_or_write_manifest(manifest, selected)  # no raise
    with pytest.raises(f.FetchError):
        f.verify_or_write_manifest(manifest, _pool([1, 3]))
