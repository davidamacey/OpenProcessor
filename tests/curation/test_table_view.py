"""`GET /table/columns` and `GET /table`: sorted, cell-exact, cursor-paged view-model rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.paging_fakes import AggregatingPagingOpenSearch, images_client
from src.config import IndexRole, index_name
from src.config.curation_accessors import base_curation_config
from src.services.curation import table_cells


if TYPE_CHECKING:
    from fastapi.testclient import TestClient

CFG = base_curation_config()
IMAGES = index_name(CFG, IndexRole.IMAGES)
ITEMS = CFG.items_index
BASE = '/curation/projects/default'
N = 23

SORTABLE = {
    'detections': [c.id for c in table_cells.DETECTION_COLUMNS if c.sortable],
    'images': [c.id for c in table_cells.IMAGE_COLUMNS if c.sortable],
}
# The wire key each detection column shows (so a cell can be compared with `GET /crops`).
WIRE_KEY = {
    'thumbnail': 'thumbnail_url',
    'class': 'class_name',
    'detector_class': 'detector_class_name',
    'detector_confidence': 'detector_confidence',
    'label_source': 'label_source',
    'validated': 'class_validated',
    'area': 'crop_area_norm',
    'rank': 'crop_rank_in_image',
    'clarity': 'blur_lap_ratio',
    'updated_at': 'updated_at',
}


def _world() -> dict[str, dict[str, dict[str, Any]]]:
    images, items = {}, {}
    for i in range(N):
        images[f'img{i:03d}'] = {
            'image_id': f'img{i:03d}',
            'image_path': f'/data/{(i * 7) % N:03d}.jpg',
            'width': 640,
            'height': 480,
            'source': 'disk',
            # ties on purpose: the image_id tiebreak has to place equal values
            'indexed_at': f'2026-01-{(i % 5) + 1:02d}T00:00:00Z',
            # the fake sorts on the flat dotted key, the card reads the nested block
            'capture': {'taken_local': f'2025-06-{(i * 3) % 28 + 1:02d}T10:00:00'},
            'capture.taken_local': f'2025-06-{(i * 3) % 28 + 1:02d}T10:00:00',
        }
        item: dict[str, Any] = {
            'crop_id': f'c{i:03d}',
            'image_id': f'img{i:03d}',
            'image_path': f'/data/{i:03d}.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'crop_area_norm': round(((i * 5) % N) / N, 3),
            'crop_rank_in_image': (i % 4) + 1,
            'detector_class_name': 'car' if i % 2 else 'bus',
            'detector_confidence': round(((i * 3) % N) / N, 3),
            'blur_lap_ratio': round(((i * 11) % N) / N, 3),
            'updated_at': f'2026-02-{(i % 9) + 1:02d}T00:00:00Z',
            'class_validated': bool(i % 3 == 0),
        }
        if i % 4:
            item['class_name'] = ['car', 'bus', 'van'][i % 3]
            item['label_source'] = 'human' if i % 3 == 0 else 'vlm'
        items[item['crop_id']] = item
    return {IMAGES: images, ITEMS: items}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return images_client(AggregatingPagingOpenSearch(_world()), monkeypatch)


def _table(client: TestClient, **params: Any) -> dict[str, Any]:
    r = client.get(f'{BASE}/table', params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _list_ids(client: TestClient, rows: str, sort: str) -> list[str]:
    if rows == 'detections':
        r = client.get(f'{BASE}/crops', params={'sort': sort, 'limit': 200})
        return [c['id'] for c in r.json()['crops']]
    r = client.get(f'{BASE}/images', params={'sort': sort, 'limit': 200})
    return [c['image_id'] for c in r.json()['items']]


def _row_id(rows: str, row: dict[str, Any]) -> str:
    return row['crop_id'] if rows == 'detections' else row['image_id']


CASES = [
    (rows, column, direction)
    for rows, ids in SORTABLE.items()
    for column in ids
    for direction in ('asc', 'desc')
]


@pytest.mark.parametrize(('rows', 'column', 'direction'), CASES)
def test_every_sortable_column_sorts_server_side(
    client: TestClient, rows: str, column: str, direction: str
) -> None:
    got = _table(client, rows=rows, sort=f'{column}:{direction}', columns=column, limit=200)
    assert got['sort_applied'] == f'{column}:{direction}'
    assert {o['id'] for o in got['sort_options']} >= {f'{column}:{direction}'}
    field = (
        table_cells.DETECTION_SORT_FIELD if rows == 'detections' else table_cells.IMAGE_SORT_FIELD
    )[column]
    served = [_row_id(rows, r) for r in got['items']]
    assert len(served) == N
    # The order is the list route's order for the same sort, not a re-sort of one page.
    assert served == _list_ids(client, rows, f'{field}:{direction}')
    values = [
        r['cells'][column]['value'] for r in got['items'] if r['cells'][column]['value'] is not None
    ]
    assert values == sorted(values, reverse=direction == 'desc'), column
    assert len(set(map(str, values))) > 1, f'{column} fixture has no spread to sort'


def test_cells_equal_the_item_wire(client: TestClient) -> None:
    columns = [c.id for c in table_cells.DETECTION_COLUMNS]
    got = _table(client, columns=','.join(columns), limit=200)
    assert [c['id'] for c in got['columns']] == columns
    wires = {c['id']: c for c in client.get(f'{BASE}/crops', params={'limit': 200}).json()['crops']}
    assert len(got['items']) == N
    for row in got['items']:
        wire = wires[row['crop_id']]
        assert set(row['cells']) == set(columns)
        for column, key in WIRE_KEY.items():
            cell = row['cells'][column]
            expected = wire.get(key)
            assert cell['value'] == (expected if expected not in ('', None) else None), column
            assert isinstance(cell['text'], str)
        assert row['image_id'] == wire['image_id']
        assert row['thumbnail_url'] == wire['thumbnail_url']
    unlabeled = next(r for r in got['items'] if r['crop_id'] == 'c000')['cells']['class']
    assert (unlabeled['text'], unlabeled['tone']) == ('Unlabeled', 'unlabeled')
    confidence = got['items'][0]['cells']['detector_confidence']
    assert confidence['text'].endswith('%')


def test_image_cells_equal_the_image_wire(client: TestClient) -> None:
    got = _table(
        client, rows='images', columns='file,items,labeled,unlabeled,indexed_at', limit=200
    )
    cards = {
        c['image_id']: c
        for c in client.get(f'{BASE}/images', params={'limit': 200}).json()['items']
    }
    assert len(got['items']) == N
    for row in got['items']:
        card = cards[row['image_id']]
        assert row['crop_id'] is None
        assert row['cells']['file']['value'] == card['file_name']
        assert row['cells']['items']['value'] == card['counts']['items']
        assert row['cells']['unlabeled']['value'] == card['counts']['unlabeled']
        assert row['cells']['indexed_at']['value'] == card['indexed_at']


@pytest.mark.parametrize('rows', ['detections', 'images'])
def test_cursor_walks_without_repeats(client: TestClient, rows: str) -> None:
    everything = [_row_id(rows, r) for r in _table(client, rows=rows, limit=200)['items']]
    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        params: dict[str, Any] = {'rows': rows, 'limit': 5}
        if cursor:
            params['cursor'] = cursor
        page = _table(client, **params)
        assert page['total'] == N
        assert page['limit'] == 5
        seen += [_row_id(rows, r) for r in page['items']]
        pages += 1
        cursor = page['next_cursor']
        if cursor is None:
            break
        assert pages <= N
    assert seen == everything
    assert len(set(seen)) == N
    assert pages == 5


def test_a_cursor_of_another_sort_is_refused(client: TestClient) -> None:
    cursor = _table(client, sort='rank:asc', limit=3)['next_cursor']
    r = client.get(f'{BASE}/table', params={'sort': 'area:asc', 'cursor': cursor})
    assert r.status_code == 400
    assert r.json()['detail']['error'] == 'cursor_invalid'


def test_unknown_column_is_refused(client: TestClient) -> None:
    r = client.get(f'{BASE}/table', params={'columns': 'class,nope'})
    assert r.status_code == 422
    assert r.json()['detail']['error'] == 'unknown_column'
    assert 'nope' in r.json()['detail']['message']
    wrong_rows = client.get(f'{BASE}/table', params={'rows': 'images', 'columns': 'class'})
    assert wrong_rows.status_code == 422


def test_an_unsortable_column_is_refused(client: TestClient) -> None:
    r = client.get(f'{BASE}/table', params={'sort': 'cluster:asc'})
    assert r.status_code == 400


def test_columns_route_serves_the_set_and_the_default(client: TestClient) -> None:
    got = client.get(f'{BASE}/table/columns').json()
    assert got['rows'] == 'detections'
    ids = [c['id'] for c in got['columns']]
    assert set(got['default']) <= set(ids)
    assert {c['id'] for c in got['columns'] if c['sortable']} == set(SORTABLE['detections'])
    images = client.get(f'{BASE}/table/columns', params={'rows': 'images'}).json()
    assert {c['id'] for c in images['columns'] if c['sortable']} == set(SORTABLE['images'])


def test_the_item_filter_narrows_the_table_like_crops(client: TestClient) -> None:
    got = _table(client, class_name='car', limit=200)
    crops = client.get(f'{BASE}/crops', params={'class_name': 'car', 'limit': 200}).json()
    assert got['total'] == crops['total'] > 0
    assert {r['crop_id'] for r in got['items']} == {c['id'] for c in crops['crops']}
    empty = _table(client, class_name='nope')
    assert (empty['items'], empty['empty_reason']) == ([], 'no_match')
    assert empty['empty_message']
