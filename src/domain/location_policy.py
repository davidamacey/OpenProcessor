"""The one visibility rule for a photo's coordinates: the project's ``location_policy``.

Every writer and every reader of a location goes through :func:`apply_location_policy`; a path that
did not would be the location leak the research spatial plan 3.1 cites (one mapper for every read
path). ``store`` keeps the fix as parsed, ``store_coarse`` keeps latitude and longitude rounded to
0.01 degree (about 1 km) and nothing finer, ``drop`` keeps no coordinate at all. An unknown policy
fails closed.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Final

from src.domain.capture_metadata import CaptureLocation, CaptureMetadata
from src.domain.geo_box import GeoBox


LOCATION_POLICIES: Final = ('drop', 'store_coarse', 'store')
DEFAULT_LOCATION_POLICY: Final = 'store'
COARSE_DECIMALS: Final = 2


def coarse_location(location: CaptureLocation) -> CaptureLocation | None:
    """``location`` rounded to :data:`COARSE_DECIMALS` decimals with altitude, accuracy and
    heading removed; ``None`` when rounding lands on (0, 0), which is no location."""
    lat = round(location.latitude, COARSE_DECIMALS)
    lon = round(location.longitude, COARSE_DECIMALS)
    if lat == 0 and lon == 0:
        return None
    return CaptureLocation(latitude=lat, longitude=lon, precision='coarse')


def apply_location_policy(meta: CaptureMetadata, policy: str) -> CaptureMetadata:
    """``meta`` as the project's ``policy`` lets it be stored or served."""
    _check_policy(policy)
    if meta.location is None or policy == 'store':
        return meta
    if policy == 'drop':
        return replace(meta, location=None)
    return replace(meta, location=coarse_location(meta.location))


def _check_policy(policy: str) -> None:
    if policy not in LOCATION_POLICIES:
        msg = f'location_policy must be one of {LOCATION_POLICIES}, not {policy!r}'
        raise ValueError(msg)


def _outward(value: float, *, up: bool) -> float:
    scaled = round(value * 10**COARSE_DECIMALS, 6)
    return (math.ceil(scaled) if up else math.floor(scaled)) / 10**COARSE_DECIMALS


def snap_box(box: GeoBox, policy: str) -> GeoBox | None:
    """``box`` as an area filter may ask it under ``policy``: coarse rounds every edge outward
    to :data:`COARSE_DECIMALS`, so a drawn area never reveals a position finer than the policy
    serves; ``None`` under ``drop`` (nothing matches)."""
    _check_policy(policy)
    if policy == 'drop':
        return None
    if policy == 'store':
        return box
    return GeoBox(
        west=max(-180.0, _outward(box.west, up=False)),
        south=max(-90.0, _outward(box.south, up=False)),
        east=min(180.0, _outward(box.east, up=True)),
        north=min(90.0, _outward(box.north, up=True)),
    )


def snap_polygon(
    vertices: list[tuple[float, float]], policy: str
) -> list[tuple[float, float]] | None:
    """``vertices`` (lon, lat) rounded to :data:`COARSE_DECIMALS` under ``store_coarse``;
    ``None`` under ``drop``."""
    _check_policy(policy)
    if policy == 'drop':
        return None
    if policy == 'store':
        return vertices
    return [(round(lon, COARSE_DECIMALS), round(lat, COARSE_DECIMALS)) for lon, lat in vertices]


__all__ = [
    'COARSE_DECIMALS',
    'DEFAULT_LOCATION_POLICY',
    'LOCATION_POLICIES',
    'apply_location_policy',
    'coarse_location',
    'snap_box',
    'snap_polygon',
]
