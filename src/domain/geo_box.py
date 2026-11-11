"""An area on the map: a degrees box and the parsers for the query spellings of an area.

``west > east`` is a box across the antimeridian (RFC 7946 section 5.2).
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, model_validator


POLYGON_MIN_VERTICES: Final = 3
POLYGON_MAX_VERTICES: Final = 64
BOX_MESSAGE: Final = 'geo_bbox needs west,south,east,north in degrees'
POLYGON_MESSAGE: Final = (
    f'geo_polygon needs {POLYGON_MIN_VERTICES} to {POLYGON_MAX_VERTICES} lon,lat vertices'
)


class GeoBox(BaseModel):
    west: float
    south: float
    east: float
    north: float

    @model_validator(mode='after')
    def _in_range(self) -> GeoBox:
        if not (
            -90 <= self.south <= self.north <= 90
            and -180 <= self.west <= 180
            and -180 <= self.east <= 180
        ):
            msg = 'box is outside -180..180 longitude and -90..90 latitude'
            raise ValueError(msg)
        return self


def parse_geo_box(raw: str, message: str = BOX_MESSAGE) -> GeoBox:
    """``west,south,east,north``; ``ValueError(message)`` for anything else."""
    try:
        west, south, east, north = (float(part) for part in raw.split(','))
        return GeoBox(west=west, south=south, east=east, north=north)
    except ValueError as exc:
        raise ValueError(message) from exc


def check_polygon(vertices: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``vertices`` when they are a usable ring; ``ValueError(POLYGON_MESSAGE)`` otherwise."""
    if not POLYGON_MIN_VERTICES <= len(vertices) <= POLYGON_MAX_VERTICES or any(
        not (-180 <= lon <= 180 and -90 <= lat <= 90) for lon, lat in vertices
    ):
        raise ValueError(POLYGON_MESSAGE)
    return vertices


def parse_geo_polygon(raw: str) -> list[tuple[float, float]]:
    """``lon,lat;lon,lat;...``; ``ValueError(POLYGON_MESSAGE)`` for anything else."""
    try:
        vertices = [
            (float(lon), float(lat))
            for lon, lat in (part.split(',') for part in raw.split(';') if part)
        ]
    except ValueError as exc:
        raise ValueError(POLYGON_MESSAGE) from exc
    return check_polygon(vertices)
