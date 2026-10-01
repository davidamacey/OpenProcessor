"""The ``sample-coco-cars`` selection (W6): CC BY car images, pinned.

The real ``coco_car_60.json`` can only be produced on a host with network
access to the COCO annotation zip, so it is generated and committed from
there; until it is, the pin test skips with that reason. The selection
logic itself is exercised here on a synthetic COCO-shaped dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.datasets import fetch_coco_subset as f


MANIFEST = Path(__file__).resolve().parents[2] / 'scripts/datasets/manifests/coco_car_60.json'
LIC_BY, LIC_BY_SA, LIC_NC = 4, 5, 1
CATEGORIES: list[dict[str, Any]] = [{'id': 3, 'name': 'car'}, {'id': 18, 'name': 'dog'}]
LICENSES: list[dict[str, Any]] = [
    {'id': LIC_NC, 'name': 'Attribution-NonCommercial-ShareAlike License'},
    {'id': LIC_BY, 'name': 'Attribution License'},
    {'id': LIC_BY_SA, 'name': 'Attribution-ShareAlike License'},
]


def _dataset() -> dict[str, Any]:
    images, anns = [], []
    for i in range(1, 31):
        lic = (LIC_BY, LIC_BY_SA, LIC_NC)[i % 3]
        images.append(
            {
                'id': i,
                'file_name': f'{i:012d}.jpg',
                'license': lic,
                'width': 640,
                'height': 480,
                'flickr_url': f'http://example.invalid/{i}',
                'coco_url': '',
            }
        )
        cat = 3 if i % 5 else 18  # every fifth image is a dog, not a car
        anns.append({'image_id': i, 'category_id': cat, 'bbox': [10, 10, 300, 300], 'iscrowd': 0})
    return {'images': images, 'annotations': anns, 'categories': CATEGORIES, 'licenses': LICENSES}


def _pick() -> list[dict]:
    data = _dataset()
    name_of: dict[int, str] = {c['id']: c['name'] for c in CATEGORIES}
    primary = f.primary_class_per_image(data['images'], data['annotations'], name_of, {3})
    pool = f.eligible_images(
        data['images'],
        primary,
        {lic['id']: lic['name'] for lic in LICENSES},
        f.resolve_license_names(['by']),
        'val2017',
    )
    return f.select_per_class(pool, [], ['car'], 60, 20260925)


@pytest.fixture
def selected() -> list[dict]:
    return _pick()


def test_only_cc_by_car_images_are_selected(selected: list[dict]) -> None:
    assert selected
    assert {r['license_name'] for r in selected} == {'Attribution License'}
    assert {r['primary_class'] for r in selected} == {'car'}
    # ShareAlike / NonCommercial images and the dog images are all excluded
    assert all(r['image_id'] % 3 == 0 and r['image_id'] % 5 for r in selected)


def test_selection_is_seed_stable(selected: list[dict]) -> None:
    assert [r['image_id'] for r in _pick()] == [r['image_id'] for r in selected]


@pytest.mark.skipif(
    not MANIFEST.is_file(),
    reason=(
        'coco_car_60.json must be generated on a host with network access '
        '(make sample-coco-cars) and committed; see the W6 deferred-live list'
    ),
)
def test_pinned_manifest_is_sixty_cc_by_cars() -> None:
    rows = json.loads(MANIFEST.read_text())
    assert len(rows) == 60
    assert {r['license_name'] for r in rows} == {'Attribution License'}
    assert {r['primary_class'] for r in rows} == {'car'}
    assert len({r['image_id'] for r in rows}) == 60
