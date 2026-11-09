"""Open Images category subsets: class names resolve to MIDs, the filter is seeded."""

from __future__ import annotations

import pytest

from scripts.datasets import fetch_openimages_bench as oi
from scripts.datasets._common import FetchError


DESCRIPTIONS = '\n'.join(f'/m/{n.lower()},{n}' for names in oi.CATEGORIES.values() for n in names)
LABELS = (
    'ImageID,Source,LabelName,Confidence\n'
    'a,human,/m/car,1\n'
    'b,human,/m/car,0\n'
    'c,human,/m/dog,1\n'
    'd,human,/m/person,1\n'
    'e,human,/m/bus,1\n'
)
IMAGES = 'ImageID,OriginalURL,License,Author\n' + ''.join(
    f'{i},u,{oi.ALLOWED_LICENSE_URL},x\n' for i in 'abcdef'
)


def _pool_for(category: str | None) -> list[dict]:
    pool = oi.eligible_rows(IMAGES)
    if category is None:
        return pool
    mids = oi.category_mids(DESCRIPTIONS, category)
    return oi.restrict_pool(pool, oi.images_with_labels(LABELS, mids))


def test_vehicle_filter_keeps_only_confirmed_vehicle_images() -> None:
    assert [r['image_id'] for r in _pool_for('vehicles')] == ['a', 'e']


def test_selection_is_deterministic_for_a_seed() -> None:
    a = oi.candidate_order(_pool_for('vehicles'), 7)
    assert a == oi.candidate_order(_pool_for('vehicles'), 7)


def test_no_category_is_the_existing_selection() -> None:
    pool = _pool_for(None)
    assert [r['image_id'] for r in pool] == list('abcdef')
    assert oi.candidate_order(pool, 3) == oi.candidate_order(oi.eligible_rows(IMAGES), 3)


def test_unknown_class_name_fails_closed() -> None:
    with pytest.raises(FetchError):
        oi.category_mids('/m/car,Car\n', 'vehicles')
    with pytest.raises(FetchError):
        oi.category_mids(DESCRIPTIONS, 'plants')


def test_pin_records_category() -> None:
    assert oi.build_pin([], 1, 'people')['category'] == 'people'
    assert 'category' not in oi.build_pin([], 1)
