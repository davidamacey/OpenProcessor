"""The project map: GeoJSON cells from a geotile_grid aggregation, under the location policy."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest
from _embedding_registry_fake import DEFAULT_WRITE_FIELDS

from src.config import CurationConfig
from src.config.curation import _tile_url
from src.domain.geo_box import parse_geo_box
from src.routers.curation import map as route
from src.services.curation import image_join, map_cells
from src.services.curation.item_filter import ItemFilter


class FakeSearch:
    """Answers the three searches the map issues and records every body."""

    def __init__(
        self,
        *,
        buckets: list[dict[str, Any]] | None = None,
        joined: list[str] | None = None,
        other_docs: int = 0,
        located: int = 3,
        unlocated: int = 4,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._buckets = buckets or []
        self._joined = joined or []
        self._other = other_docs
        self._located = located
        self._unlocated = unlocated

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((index, body))
        aggs = body.get('aggs') or {}
        if 'cells' in aggs:
            return {'aggregations': {'cells': {'buckets': self._buckets}}}
        if 'images' in aggs:
            return {
                'aggregations': {
                    'images': {
                        'buckets': [{'key': i, 'doc_count': 1} for i in self._joined],
                        'sum_other_doc_count': self._other,
                    }
                }
            }
        if 'extent' in aggs:
            return {
                'hits': {'total': {'value': self._located}},
                'aggregations': {
                    'extent': {
                        'bounds': {
                            'top_left': {'lat': 48.0, 'lon': 7.0},
                            'bottom_right': {'lat': 46.0, 'lon': 9.0},
                        }
                    }
                },
            }
        if body.get('size') == 0 and 'must_not' in json.dumps(body.get('query')):
            return {'hits': {'total': {'value': self._unlocated}, 'hits': []}}
        ids = self._joined or ['i1', 'i2']
        return {'hits': {'total': {'value': 250}, 'hits': [{'_id': i} for i in ids]}}


def _bucket(key: str, count: int, lon: float, lat: float, pin: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        'key': key,
        'doc_count': count,
        'centre': {'location': {'lon': lon, 'lat': lat}},
    }
    if pin is not None:
        out['pin'] = {'hits': {'hits': [{'_id': pin}]}}
    return out


BUCKETS = [_bucket('9/267/179', 12, 8.123456789, 47.987654321), _bucket('9/268/179', 1, 9.5, 47.5)]


async def _cells(os: FakeSearch, policy: str, f: ItemFilter | None = None, **kw: Any) -> Any:
    return await map_cells.map_cells(
        os,
        policy,
        f or ItemFilter(),
        DEFAULT_WRITE_FIELDS,
        zoom=kw.get('zoom', 6),
        box=kw.get('box'),
    )


@pytest.mark.asyncio
async def test_cells_are_a_geojson_feature_collection_with_lon_lat_points() -> None:
    os = FakeSearch(buckets=BUCKETS)
    got = (await _cells(os, 'store')).model_dump(mode='json')

    assert got['type'] == 'FeatureCollection'
    assert got['precision'] == 9
    first = got['features'][0]
    assert first['type'] == 'Feature'
    assert first['geometry'] == {'type': 'Point', 'coordinates': [8.123457, 47.987654]}
    props = first['properties']
    assert (props['count'], props['tile']) == (12, '9/267/179')
    index, body = os.calls[0]
    grid = body['aggs']['cells']['geotile_grid']
    assert (grid['field'], grid['precision']) == ('capture.location', 9)
    assert body['aggs']['cells']['aggs']['centre'] == {
        'geo_centroid': {'field': 'capture.location'}
    }
    assert body['query']['bool']['filter'] == [{'exists': {'field': 'capture.location'}}]
    assert index.endswith('images')


@pytest.mark.asyncio
async def test_drop_serves_nothing_and_asks_the_index_nothing() -> None:
    os = FakeSearch(buckets=BUCKETS)
    got = await _cells(os, 'drop')
    assert got.features == []
    assert os.calls == []
    meta = await map_cells.map_meta(os, 'drop', tile_url='u/{z}/{x}/{y}', tile_attribution='a')
    assert (meta.available, meta.n_located, meta.bounds) == (False, 0, None)


@pytest.mark.asyncio
async def test_store_coarse_caps_the_cell_size_and_rounds_the_markers() -> None:
    os = FakeSearch(buckets=BUCKETS)
    got = await _cells(os, 'store_coarse', zoom=20)
    assert got.precision == map_cells.COARSE_MAX_PRECISION
    assert os.calls[0][1]['aggs']['cells']['geotile_grid']['precision'] == 15
    assert got.features[0].geometry.coordinates == (8.12, 47.99)
    full = await _cells(os, 'store', zoom=20)
    assert full.precision == map_cells.MAX_PRECISION


@pytest.mark.asyncio
async def test_a_box_limits_the_query_and_the_grid_and_may_cross_the_antimeridian() -> None:
    os = FakeSearch(buckets=BUCKETS)
    box = parse_geo_box('170,-10,-170,10')
    await _cells(os, 'store', box=box)
    _, body = os.calls[0]
    corner = {
        'top_left': {'lat': 10.0, 'lon': 170.0},
        'bottom_right': {'lat': -10.0, 'lon': -170.0},
    }
    assert {'geo_bounding_box': {'capture.location': corner}} in body['query']['bool']['filter']
    assert body['aggs']['cells']['geotile_grid']['bounds'] == corner


@pytest.mark.parametrize('raw', ['1,2,3', 'a,b,c,d', '0,-91,10,10', '0,10,10,5', '-181,0,10,10'])
def test_a_bad_bbox_is_refused(raw: str) -> None:
    with pytest.raises(ValueError, match='geo_bbox needs'):
        parse_geo_box(raw)


@pytest.mark.asyncio
async def test_a_large_project_ships_a_capped_number_of_cells() -> None:
    many = [_bucket(f'13/{n}/1', 1, 8.0, 47.0) for n in range(map_cells.MAX_CELLS + 1)]
    got = await _cells(FakeSearch(buckets=many), 'store')
    assert len(got.features) == map_cells.MAX_CELLS
    assert got.truncated is True
    os = FakeSearch(buckets=many)
    await _cells(os, 'store')
    assert os.calls[0][1]['aggs']['cells']['geotile_grid']['size'] == map_cells.MAX_CELLS + 1


@pytest.mark.asyncio
async def test_a_date_range_is_answered_on_the_images_index_without_the_item_join() -> None:
    os = FakeSearch(buckets=BUCKETS)
    await _cells(
        os, 'store', ItemFilter(taken_from=dt.date(2024, 3, 5), taken_to=dt.date(2024, 3, 6))
    )
    assert len(os.calls) == 1
    flt = os.calls[0][1]['query']['bool']['filter']
    assert any('should' in c.get('bool', {}) for c in flt)
    assert 'capture.taken_utc' in json.dumps(flt)


@pytest.mark.asyncio
async def test_an_item_filter_is_resolved_to_images_first_and_says_when_it_was_cut() -> None:
    os = FakeSearch(buckets=BUCKETS, joined=['img1', 'img2'], other_docs=5)
    got = await _cells(os, 'store', ItemFilter(class_names=['car']))
    assert got.join_truncated is True
    (items_index, items_body), (_, images_body) = os.calls
    assert items_index.endswith('items')
    assert items_body['aggs']['images']['terms'] == {
        'field': 'image_id',
        'size': image_join.JOIN_MAX_IMAGES,
    }
    assert {'terms': {'image_id': ['img1', 'img2']}} in images_body['query']['bool']['filter']


@pytest.mark.asyncio
async def test_the_meta_gate_reports_extent_and_tile_settings() -> None:
    meta = await map_cells.map_meta(
        FakeSearch(located=3), 'store', tile_url='https://t/{z}/{x}/{y}.png', tile_attribution='osm'
    )
    assert meta.available is True
    assert meta.bounds is not None
    assert (meta.bounds.west, meta.bounds.north) == (7.0, 48.0)
    assert (meta.tile_url, meta.tile_attribution) == ('https://t/{z}/{x}/{y}.png', 'osm')
    none = await map_cells.map_meta(
        FakeSearch(located=0), 'store', tile_url='u', tile_attribution='a'
    )
    assert (none.available, none.bounds) == (False, None)


def test_a_tile_key_is_a_box_in_degrees() -> None:
    world = map_cells.tile_box('0/0/0')
    assert (world.west, world.east) == (-180.0, 180.0)
    assert world.north == pytest.approx(85.0511, abs=1e-3)
    assert map_cells.tile_box('1/1/0').west == 0.0
    for bad in ('3/9/0', 'a/b/c', '1/2', '23/0/0'):
        with pytest.raises(map_cells.MapQueryError):
            map_cells.tile_box(bad)


@pytest.mark.asyncio
async def test_the_route_answers_geo_json_and_a_bad_box_is_a_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    async def policy() -> str:
        return 'store'

    async def write_fields() -> Any:
        return DEFAULT_WRITE_FIELDS

    monkeypatch.setattr(route, 'get_location_policy', policy)
    monkeypatch.setattr(route, 'item_write_fields', write_fields)
    response = await route.get_map_cells(
        FakeSearch(buckets=BUCKETS), ItemFilter(), zoom=6, bbox=None
    )
    assert response.media_type == 'application/geo+json'
    assert json.loads(response.body)['type'] == 'FeatureCollection'
    with pytest.raises(HTTPException) as bad:
        await route.get_map_cells(FakeSearch(), ItemFilter(), zoom=6, bbox='1,2')
    assert bad.value.status_code == 400


def test_the_tile_template_is_validated_and_defaults_to_openstreetmap() -> None:
    config = CurationConfig()
    assert config.map_tile_url == 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
    assert 'OpenStreetMap' in config.map_tile_attribution
    assert _tile_url('https://tiles.lan/{z}/{x}/{y}.png') == 'https://tiles.lan/{z}/{x}/{y}.png'
    with pytest.raises(ValueError, match='OP_MAP_TILE_URL'):
        _tile_url('https://tiles.lan/tiles.png')


@pytest.mark.asyncio
async def test_cell_serves_its_bounds() -> None:
    got = (await _cells(FakeSearch(buckets=BUCKETS), 'store')).model_dump(mode='json')
    served = got['features'][0]['properties']['bounds']
    assert served == map_cells.tile_box('9/267/179').model_dump()
    coarse = (await _cells(FakeSearch(buckets=BUCKETS), 'store_coarse')).model_dump(mode='json')
    # Rounded outward to 0.01 degree, never finer than the policy serves.
    edge = coarse['features'][0]['properties']['bounds']
    assert all(round(v, 2) == v for v in edge.values())
    box = map_cells.tile_box('9/267/179')
    assert edge['west'] <= box.west
    assert edge['east'] >= box.east


@pytest.mark.asyncio
async def test_single_photo_cell_at_max_precision_serves_its_image() -> None:
    lone = [_bucket('22/1/1', 1, 8.0, 47.0, pin='img-9'), _bucket('22/2/1', 3, 8.1, 47.0, pin='x')]
    os = FakeSearch(buckets=lone)
    got = (await _cells(os, 'store', zoom=19)).model_dump(mode='json')
    one, many = (f['properties'] for f in got['features'])
    assert one['image_id'] == 'img-9'
    assert one['thumbnail_url'].endswith('/images/img-9/thumbnail')
    assert (many['image_id'], many['thumbnail_url']) == (None, None)
    assert 'pin' in os.calls[0][1]['aggs']['cells']['aggs']
    coarse_os = FakeSearch(buckets=BUCKETS)
    await _cells(coarse_os, 'store', zoom=6)
    assert 'pin' not in coarse_os.calls[0][1]['aggs']['cells']['aggs']


@pytest.mark.asyncio
async def test_meta_counts_images_without_location() -> None:
    meta = await map_cells.map_meta(
        FakeSearch(located=3, unlocated=4), 'store', tile_url='u', tile_attribution='a'
    )
    assert meta.n_without_location == 4
    dropped = await map_cells.map_meta(
        FakeSearch(unlocated=9), 'drop', tile_url='u', tile_attribution='a'
    )
    # Under `drop` no image has a served position.
    assert (dropped.n_located, dropped.n_without_location) == (0, 250)
