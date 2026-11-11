"""The project map: images with a stored position, aggregated into grid cells on the server.

OpenSearch holds each image's position as a `geo_point` (`capture.location`), so the browser never
receives the points of a large project: `geotile_grid` buckets them into web-mercator tiles at the
map's zoom, `geo_centroid` places each cell's marker, and the response is capped (5000 cells), so a
1M-image project ships a few thousand features. A box (`geo_bounding_box`, antimeridian aware) limits
the work to what is on screen.

* geotile_grid: https://docs.opensearch.org/latest/aggregations/bucket/geotile-grid/
* geo_bounding_box: https://docs.opensearch.org/latest/query-dsl/geo-and-xy/geo-bounding-box/
* geo_centroid: https://docs.opensearch.org/latest/aggregations/metric/geocentroid/

The policy applies at read time as well as at write time: under `drop` nothing is ever served (the
purge may not have run yet), under `store_coarse` cells are no finer than about 1 km and the markers
are rounded to 0.01 degree. Item filters cannot be applied to the images index (no coordinate is
ever copied to an item), so any item-level predicate is resolved to the images it matches first
(at most `JOIN_MAX_IMAGES` of them, `join_truncated` says when it was more).
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from src.config.curation_accessors import images_index, items_index
from src.domain.location_policy import COARSE_DECIMALS
from src.schemas.curation.map import (
    MapBounds,
    MapCellCollection,
    MapCellFeature,
    MapCellGeometry,
    MapCellProperties,
    MapMeta,
    MapTileImages,
)
from src.services.curation.item_filter import ItemFilter, item_filter_query


if TYPE_CHECKING:
    from src.services.embedding_spaces.readers import WriteFields

LOCATION_FIELD: Final = 'capture.location'
MAX_CELLS: Final = 5000
#: Cells per tile at the requested zoom: a map tile is 256 px, a cell 2^3 times finer than a tile.
CELL_ZOOM_OFFSET: Final = 3
MAX_PRECISION: Final = 22
#: A coarse policy keeps about 1 km: tiles at zoom 15 are about 1.2 km wide at the equator.
COARSE_MAX_PRECISION: Final = 15
JOIN_MAX_IMAGES: Final = 10_000
TILE_IMAGES_MAX: Final = 200


class MapQueryError(ValueError):
    """The request names a box or tile that does not exist."""


def max_precision(policy: str) -> int:
    return COARSE_MAX_PRECISION if policy == 'store_coarse' else MAX_PRECISION


def parse_bbox(raw: str) -> MapBounds:
    """``west,south,east,north`` in degrees; west may exceed east (a box across the antimeridian)."""
    try:
        west, south, east, north = (float(part) for part in raw.split(','))
    except ValueError as exc:
        msg = 'bbox must be west,south,east,north in degrees'
        raise MapQueryError(msg) from exc
    if not (-90 <= south <= north <= 90 and -180 <= west <= 180 and -180 <= east <= 180):
        msg = 'bbox is outside -180..180 longitude and -90..90 latitude'
        raise MapQueryError(msg)
    return MapBounds(west=west, south=south, east=east, north=north)


def geo_box_query(box: MapBounds) -> dict[str, Any]:
    return {
        'geo_bounding_box': {
            LOCATION_FIELD: {
                'top_left': {'lat': box.north, 'lon': box.west},
                'bottom_right': {'lat': box.south, 'lon': box.east},
            }
        }
    }


def _date_clauses(f: ItemFilter) -> list[dict[str, Any]]:
    """``taken_from`` / ``taken_to`` on the images index (the item copies' twin fields)."""
    if f.taken_from is None and f.taken_to is None:
        return []
    bounds: dict[str, str] = {}
    if f.taken_from is not None:
        bounds['gte'] = f'{f.taken_from.isoformat()}T00:00:00'
    if f.taken_to is not None:
        bounds['lt'] = f'{(f.taken_to + dt.timedelta(days=1)).isoformat()}T00:00:00'
    return [
        {
            'bool': {
                'should': [
                    {'range': {'capture.taken_utc': bounds}},
                    {
                        'bool': {
                            'must': [{'range': {'capture.taken_local': bounds}}],
                            'must_not': [{'exists': {'field': 'capture.taken_utc'}}],
                        }
                    },
                ],
                'minimum_should_match': 1,
            }
        }
    ]


async def _joined_image_ids(
    opensearch: Any, f: ItemFilter, embedding: WriteFields
) -> tuple[list[str], bool]:
    """The images that have an item matching ``f`` and whether there were more than we use."""
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 0,
            'query': item_filter_query(f, embedding=embedding),
            'aggs': {'images': {'terms': {'field': 'image_id', 'size': JOIN_MAX_IMAGES}}},
        },
    )
    agg = resp['aggregations']['images']
    return [b['key'] for b in agg['buckets']], int(agg.get('sum_other_doc_count', 0)) > 0


async def image_query(
    opensearch: Any, f: ItemFilter, embedding: WriteFields, box: MapBounds | None
) -> tuple[dict[str, Any], bool]:
    """The images-index query for a map request and whether the item join was truncated.

    A filter that only names capture dates is answered on the images index directly; any other
    predicate goes through the items that match it."""
    filters: list[dict[str, Any]] = [{'exists': {'field': LOCATION_FIELD}}]
    if box is not None:
        filters.append(geo_box_query(box))
    joined = f.model_copy(update={'taken_from': None, 'taken_to': None})
    truncated = False
    if joined.is_empty():
        filters.extend(_date_clauses(f))
    else:
        ids, truncated = await _joined_image_ids(opensearch, f, embedding)
        filters.append({'terms': {'image_id': ids}})
    return {'bool': {'filter': filters}}, truncated


def _round_coarse(value: float) -> float:
    return round(value, COARSE_DECIMALS)


async def map_meta(
    opensearch: Any, policy: str, *, tile_url: str, tile_attribution: str
) -> MapMeta:
    """The page's gate: nothing is available under `drop`, or without a stored position."""
    if policy == 'drop':
        return MapMeta(
            available=False,
            location_policy='drop',
            n_located=0,
            tile_url=tile_url,
            tile_attribution=tile_attribution,
            max_zoom=0,
        )
    resp = await opensearch.search(
        index=images_index(),
        body={
            'size': 0,
            'track_total_hits': True,
            'query': {'exists': {'field': LOCATION_FIELD}},
            'aggs': {'extent': {'geo_bounds': {'field': LOCATION_FIELD}}},
        },
    )
    n_located = int(resp['hits']['total']['value'])
    bounds = None
    box = (resp.get('aggregations') or {}).get('extent', {}).get('bounds')
    if n_located and box:
        bounds = MapBounds(
            west=box['top_left']['lon'],
            south=box['bottom_right']['lat'],
            east=box['bottom_right']['lon'],
            north=box['top_left']['lat'],
        )
    return MapMeta(
        available=n_located > 0,
        location_policy=cast('Literal["drop", "store_coarse", "store"]', policy),
        n_located=n_located,
        bounds=bounds,
        tile_url=tile_url,
        tile_attribution=tile_attribution,
        max_zoom=max_precision(policy) - CELL_ZOOM_OFFSET,
    )


async def map_cells(
    opensearch: Any,
    policy: str,
    f: ItemFilter,
    embedding: WriteFields,
    *,
    zoom: int,
    box: MapBounds | None,
) -> MapCellCollection:
    """The grid cells of the matching images at ``zoom`` as a GeoJSON FeatureCollection."""
    precision = min(zoom + CELL_ZOOM_OFFSET, max_precision(policy))
    if policy == 'drop':
        return MapCellCollection(features=[], precision=precision)
    query, join_truncated = await image_query(opensearch, f, embedding, box)
    grid: dict[str, Any] = {'field': LOCATION_FIELD, 'precision': precision, 'size': MAX_CELLS + 1}
    if box is not None:
        grid['bounds'] = {
            'top_left': {'lat': box.north, 'lon': box.west},
            'bottom_right': {'lat': box.south, 'lon': box.east},
        }
    resp = await opensearch.search(
        index=images_index(),
        body={
            'size': 0,
            'query': query,
            'aggs': {
                'cells': {
                    'geotile_grid': grid,
                    'aggs': {'centre': {'geo_centroid': {'field': LOCATION_FIELD}}},
                }
            },
        },
    )
    buckets = resp['aggregations']['cells']['buckets']
    features: list[MapCellFeature] = []
    for bucket in buckets[:MAX_CELLS]:
        centre = bucket['centre']['location']
        lon, lat = float(centre['lon']), float(centre['lat'])
        if policy == 'store_coarse':
            lon, lat = _round_coarse(lon), _round_coarse(lat)
        features.append(
            MapCellFeature(
                geometry=MapCellGeometry(coordinates=(round(lon, 6), round(lat, 6))),
                properties=MapCellProperties(
                    count=int(bucket['doc_count']), tile=str(bucket['key'])
                ),
            )
        )
    return MapCellCollection(
        features=features,
        precision=precision,
        truncated=len(buckets) > MAX_CELLS,
        join_truncated=join_truncated,
    )


def tile_box(tile: str) -> MapBounds:
    """The degrees box of a `z/x/y` web-mercator tile."""
    import math

    try:
        z, x, y = (int(part) for part in tile.split('/'))
    except ValueError as exc:
        msg = 'tile must be z/x/y'
        raise MapQueryError(msg) from exc
    if not (0 <= z <= MAX_PRECISION and 0 <= x < 2**z and 0 <= y < 2**z):
        msg = f'tile {tile!r} does not exist'
        raise MapQueryError(msg)
    n = 2**z

    def lat(row: int) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * row / n))))

    return MapBounds(
        west=x / n * 360 - 180, east=(x + 1) / n * 360 - 180, north=lat(y), south=lat(y + 1)
    )


async def tile_images(
    opensearch: Any, policy: str, f: ItemFilter, embedding: WriteFields, tile: str
) -> MapTileImages:
    """The images in one grid cell, to open the review list on them (at most 200)."""
    box = tile_box(tile)
    if policy == 'drop':
        return MapTileImages(tile=tile, count=0, image_ids=[], truncated=False)
    query, _ = await image_query(opensearch, f, embedding, box)
    resp = await opensearch.search(
        index=images_index(),
        body={
            'size': TILE_IMAGES_MAX,
            'track_total_hits': True,
            'query': query,
            '_source': False,
            'sort': [{'image_id': 'asc'}],
        },
    )
    total = int(resp['hits']['total']['value'])
    ids = [hit['_id'] for hit in resp['hits']['hits']]
    return MapTileImages(tile=tile, count=total, image_ids=ids, truncated=total > len(ids))


__all__ = [
    'CELL_ZOOM_OFFSET',
    'JOIN_MAX_IMAGES',
    'MAX_CELLS',
    'MapQueryError',
    'map_cells',
    'map_meta',
    'parse_bbox',
    'tile_box',
    'tile_images',
]
