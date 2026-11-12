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


def _big_client(monkeypatch: pytest.MonkeyPatch, n: int) -> Any:
    docs = {
        f'c{i:05d}': _doc(i, confidence=round((i % 97) / 97, 4), class_name=f'k{i % 5}')
        for i in range(n)
    }
    return crops_client(PagingOpenSearch({ITEMS: docs}), monkeypatch)


def test_reads_past_ten_thousand(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 10_050
    client = _big_client(monkeypatch, n)
    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        params: dict[str, Any] = {'limit': 500, 'sort': 'confidence:asc'}
        if cursor:
            params['cursor'] = cursor
        r = client.get(f'{BASE}/crops', params=params)
        assert r.status_code == 200, r.text
        body = r.json()
        assert 'page' not in body
        assert body['total'] == n
        seen += _ids(body)
        cursor = body['next_cursor']
        pages += 1
        if cursor is None:
            break
    assert pages == 21
    assert len(seen) == len(set(seen)) == n


def test_cursor_of_another_sort_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _big_client(monkeypatch, 30)
    first = client.get(f'{BASE}/crops', params={'limit': 5, 'sort': 'confidence:asc'}).json()
    for sort in ('confidence:desc', 'class_name:asc'):
        r = client.get(
            f'{BASE}/crops', params={'limit': 5, 'sort': sort, 'cursor': first['next_cursor']}
        )
        assert r.status_code == 400
        assert r.json()['detail']['error'] == 'cursor_invalid'
    assert client.get(f'{BASE}/crops', params={'cursor': '!!'}).status_code == 400


def test_a_sort_with_missing_values_pages_without_repeats(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {
        f'c{i:05d}': _doc(i, **({'class_name': f'k{i % 3}'} if i % 4 else {})) for i in range(40)
    }
    client = crops_client(PagingOpenSearch({ITEMS: docs}), monkeypatch)
    seen: list[str] = []
    cursor = None
    while True:
        params: dict[str, Any] = {'limit': 7, 'sort': 'class_name:asc'}
        if cursor:
            params['cursor'] = cursor
        body = client.get(f'{BASE}/crops', params=params).json()
        seen += _ids(body)
        cursor = body['next_cursor']
        if cursor is None:
            break
    assert len(seen) == len(set(seen)) == 40


def test_an_empty_list_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    client = crops_client(PagingOpenSearch({ITEMS: {}}), monkeypatch)
    plain = client.get(f'{BASE}/crops').json()
    assert (plain['empty_reason'], plain['next_cursor']) == ('no_items', None)
    assert plain['empty_message']
    narrowed = client.get(f'{BASE}/crops', params={'class_name': 'nothing'}).json()
    assert narrowed['empty_reason'] == 'no_match'
