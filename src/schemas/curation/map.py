"""Wire models of the project map: `GET .../map` and `.../map/cells`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.domain.geo_box import GeoBox


class MapMeta(BaseModel):
    """What the Map page needs before it draws. `available` is false (and the page is hidden)
    when the project's location policy is `drop` or no image has a stored position."""

    available: bool
    location_policy: Literal['drop', 'store_coarse', 'store']
    n_located: int = Field(description='Images with a stored position.')
    n_without_location: int = Field(
        description='Images with no stored position, or one the location policy does not serve.'
    )
    bounds: GeoBox | None = Field(
        default=None, description='The box around every stored position; null when there is none.'
    )
    tile_url: str = Field(
        description='Raster tile template ({z}/{x}/{y}); the browser fetches tiles from it. '
        '`OP_MAP_TILE_URL`.'
    )
    tile_attribution: str
    max_zoom: int = Field(
        description='The deepest map zoom with finer cells than the policy allows nowhere.'
    )


class MapCellGeometry(BaseModel):
    type: Literal['Point'] = 'Point'
    coordinates: tuple[float, float] = Field(description='[longitude, latitude] (RFC 7946).')


class MapCellProperties(BaseModel):
    count: int = Field(description='Images in the cell.')
    tile: str = Field(description='The cell as `z/x/y` (web mercator tile).')
    bounds: GeoBox = Field(
        description='The cell as a degrees box, rounded outward under a coarse location policy.'
    )
    image_id: str | None = Field(
        default=None,
        description='The photo, when the cell holds exactly one at the finest zoom the policy allows.',
    )
    thumbnail_url: str | None = Field(default=None, description='Thumbnail of `image_id`.')


class MapCellFeature(BaseModel):
    type: Literal['Feature'] = 'Feature'
    geometry: MapCellGeometry
    properties: MapCellProperties


class MapCellCollection(BaseModel):
    """A GeoJSON FeatureCollection (RFC 7946): one Point per grid cell, at the centroid of the
    images in it. `truncated` is true when more cells matched than the response carries (zoom
    in); `join_truncated` when an item filter matched more images than the join could use."""

    type: Literal['FeatureCollection'] = 'FeatureCollection'
    features: list[MapCellFeature]
    precision: int = Field(description='The grid precision (tile zoom) the cells are at.')
    truncated: bool = False
    join_truncated: bool = False
    n_located_matching: int = Field(
        default=0,
        description='Photos with a served position that match the request filter, across the '
        'whole map (the viewport box does not narrow it).',
    )
    n_unlocated_matching: int = Field(
        default=0,
        description='Photos without a position that match the request filter: they are not on '
        'the map.',
    )
