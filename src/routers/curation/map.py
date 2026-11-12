"""Curation router sub-module: the project map.

``GET /map`` says whether the Map view is available and gives its tile settings;
``GET /map/cells`` serves the matching images as a GeoJSON FeatureCollection of grid cells at a
zoom (``application/geo+json``, RFC 7946). Both take the shared item filter (``ItemFilterQuery``), including
``taken_from`` / ``taken_to``, and follow the project's ``location_policy``: under ``drop`` nothing
is served, under ``store_coarse`` cells are no finer than about 1 km.
"""

from typing import Annotated, Any

from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse

from src.config import get_curation_config
from src.domain.geo_box import parse_geo_box
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._item_filter_params import ItemFilterQuery
from src.schemas.curation.map import MapCellCollection, MapMeta
from src.services.curation import map_cells
from src.services.curation.location_policy_store import get_location_policy
from src.services.embedding_spaces.readers import item_write_fields


GEOJSON = 'application/geo+json'


def _bad_request(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@router.get('/map', response_model=MapMeta, tags=['Map'])
async def get_map(opensearch: OpenSearchDep) -> MapMeta:
    """Whether the project has a map (a stored position and a policy that allows it), the box
    around its positions and the tile template the browser draws with."""
    config = get_curation_config()
    return await map_cells.map_meta(
        opensearch,
        await get_location_policy(),
        tile_url=config.map_tile_url,
        tile_attribution=config.map_tile_attribution,
    )


@router.get(
    '/map/cells',
    response_model=None,
    responses={200: {'model': MapCellCollection, 'content': {GEOJSON: {}}}},
    tags=['Map'],
)
async def get_map_cells(
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    zoom: Annotated[
        int, Query(ge=0, le=22, description='The map zoom; cells are finer at higher zoom.')
    ],
    bbox: Annotated[
        str | None,
        Query(
            description='west,south,east,north in degrees; west > east crosses the antimeridian.'
        ),
    ] = None,
) -> Any:
    """Grid cells of the images that have a position, as GeoJSON Points at each cell's centroid.
    Capped at 5000 cells (`truncated`: zoom in)."""
    try:
        box = (
            None
            if bbox is None
            else parse_geo_box(bbox, 'bbox needs west,south,east,north in degrees')
        )
        collection = await map_cells.map_cells(
            opensearch,
            await get_location_policy(),
            item_filter,
            await item_write_fields(),
            zoom=zoom,
            box=box,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return JSONResponse(collection.model_dump(mode='json'), media_type=GEOJSON)
