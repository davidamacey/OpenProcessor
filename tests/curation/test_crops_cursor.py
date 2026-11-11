"""`GET /crops` sorts: the served options and the class, detector class and validated orders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from curation.paging_fakes import PagingOpenSearch, crops_client
from src.config.curation_accessors import base_curation_config
from src.services.curation.crop_browse import CROP_SORT_FIELDS, CROP_SORT_LABELS


if TYPE_CHECKING:
    import pytest

ITEMS = base_curation_config().items_index
BASE = '/curation/projects/default'


def _doc(n: int, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': f'c{n:05d}',
        'image_id': f'img-{n}',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        **extra,
    }


def _ids(body: dict[str, Any]) -> list[str]:
    return [c['id'] for c in body['crops']]


def test_every_sort_field_labels_both_directions() -> None:
    assert set(CROP_SORT_LABELS) == set(CROP_SORT_FIELDS)
    for labels in CROP_SORT_LABELS.values():
        assert set(labels) == {'asc', 'desc'}
        assert all(labels.values())


def test_sort_options_are_served_with_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    client = crops_client(PagingOpenSearch({ITEMS: {'a': _doc(1)}}), monkeypatch)
    body = client.get(f'{BASE}/crops').json()
    assert body['sort_applied'] == 'updated_at:desc'
    options = {o['id']: o['label'] for o in body['sort_options']}
    assert len(options) == 2 * len(CROP_SORT_FIELDS)
    assert options['detector_confidence:asc'] == 'Detector confidence (low first)'
    assert {'class_name:asc', 'detector_class_name:desc', 'validated:desc'} <= set(options)
    explicit = client.get(f'{BASE}/crops', params={'sort': 'confidence'}).json()
    assert explicit['sort_applied'] == 'confidence:desc'


def test_sort_by_class_name_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {
        'c1': _doc(1, class_name='truck', detector_class_name='bus', class_validated=True),
        'c2': _doc(2, class_name='bike', detector_class_name='car', class_validated=False),
        'c3': _doc(3),
        'c4': _doc(4, class_name='car', detector_class_name='bike', class_validated=True),
    }
    client = crops_client(PagingOpenSearch({ITEMS: docs}), monkeypatch)

    def ids(sort: str) -> list[str]:
        r = client.get(f'{BASE}/crops', params={'sort': sort})
        assert r.status_code == 200, r.text
        return _ids(r.json())

    assert ids('class_name:asc') == ['c00002', 'c00004', 'c00001', 'c00003']
    assert ids('class_name:desc') == ['c00001', 'c00004', 'c00002', 'c00003']
    assert ids('detector_class_name:asc') == ['c00004', 'c00001', 'c00002', 'c00003']
    assert ids('validated:desc') == ['c00001', 'c00004', 'c00002', 'c00003']
    assert ids('validated:asc') == ['c00002', 'c00001', 'c00004', 'c00003']
