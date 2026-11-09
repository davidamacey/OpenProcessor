"""The capture fields: ``capture`` on the images index, two copies on items.

``capture.taken_local`` is wall-clock time with no zone, stored as floating time (as if UTC), for
local-hour and day views; ``capture.taken_utc`` is set only when the offset is known. Items carry
``capture_taken_local`` and ``capture_taken_utc`` so a crops query filters on capture date without
a join. **No coordinate is ever copied to an item**: ``capture.location`` (a ``geo_point``) exists
only on the images index, and only when the project's ``location_policy`` allows it
(``src/domain/location_policy.py``).

Additive and idempotent: :func:`ensure_capture_fields` migrates indexes created before the fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.domain.capture_metadata import CaptureMetadata

logger = get_logger(__name__)

_FLOATING_TIME = 'strict_date_hour_minute_second_fraction||strict_date_hour_minute_second'

CAPTURE_IMAGE_MAPPING: dict[str, Any] = {
    'capture': {
        'properties': {
            'taken_local': {'type': 'date', 'format': _FLOATING_TIME},
            'taken_utc': {'type': 'date', 'format': _FLOATING_TIME},
            'offset_minutes': {'type': 'short'},
            'time_source': {'type': 'keyword'},
            'offset_source': {'type': 'keyword'},
            'time_precision': {'type': 'keyword'},
            'camera_make': {'type': 'keyword'},
            'camera_model': {'type': 'keyword'},
            'lens_model': {'type': 'keyword'},
            'focal_length_mm': {'type': 'float'},
            'focal_length_35mm': {'type': 'float'},
            'orientation': {'type': 'byte'},
            'has_location': {'type': 'boolean'},
            'exif_status': {'type': 'keyword'},
            'parser': {'type': 'keyword'},
            'location': {'type': 'geo_point'},
            'altitude_m': {'type': 'float'},
            'location_accuracy_m': {'type': 'float'},
            'heading_deg': {'type': 'float'},
            'location_precision': {'type': 'keyword'},
        }
    }
}

CAPTURE_ITEM_MAPPING: dict[str, Any] = {
    'capture_taken_local': {'type': 'date', 'format': _FLOATING_TIME},
    'capture_taken_utc': {'type': 'date', 'format': _FLOATING_TIME},
}

#: The fields under ``capture`` that carry a position (what a location purge clears).
LOCATION_FIELDS = (
    'location',
    'altitude_m',
    'location_accuracy_m',
    'heading_deg',
    'location_precision',
)


def capture_image_block(meta: CaptureMetadata) -> dict[str, Any]:
    """The ``capture`` object of an images document. ``meta`` has already been through
    ``apply_location_policy``: whatever location it still holds is stored."""
    block: dict[str, Any] = {
        'exif_status': meta.exif_status,
        'has_location': meta.has_location,
        'time_source': meta.time_source,
        'offset_source': meta.offset_source,
    }
    optional: dict[str, Any] = {
        'parser': meta.parser,
        'taken_local': meta.taken_local,
        'taken_utc': meta.taken_utc,
        'offset_minutes': meta.offset_minutes,
        'time_precision': meta.time_precision,
        'camera_make': meta.camera_make,
        'camera_model': meta.camera_model,
        'lens_model': meta.lens_model,
        'focal_length_mm': meta.focal_length_mm,
        'focal_length_35mm': meta.focal_length_35mm,
        'orientation': meta.orientation,
    }
    block.update({k: v for k, v in optional.items() if v is not None})
    loc = meta.location
    if loc is not None:
        block['location'] = {'lat': loc.latitude, 'lon': loc.longitude}
        block['location_precision'] = loc.precision
        extras = {
            'altitude_m': loc.altitude_m,
            'location_accuracy_m': loc.accuracy_m,
            'heading_deg': loc.heading_deg,
        }
        block.update({k: v for k, v in extras.items() if v is not None})
    return block


def capture_item_fields(meta: CaptureMetadata) -> dict[str, Any]:
    """The two capture-time copies an item document carries (never a coordinate)."""
    fields = {
        'capture_taken_local': meta.taken_local,
        'capture_taken_utc': meta.taken_utc,
    }
    return {k: v for k, v in fields.items() if v is not None}


async def ensure_capture_fields(client: AsyncOpenSearch) -> None:
    """PUT the capture mappings onto the bound project's images and items indexes."""
    from src.clients.curation_opensearch.base import _is_recoverable_mapping_conflict
    from src.config import get_curation_config

    config = get_curation_config()
    for index, properties in (
        (config.images_index, CAPTURE_IMAGE_MAPPING),
        (config.items_index, CAPTURE_ITEM_MAPPING),
    ):
        try:
            await client.indices.put_mapping(index=index, body={'properties': properties})
        except Exception as exc:
            if not _is_recoverable_mapping_conflict(str(exc)):
                raise
    logger.info('curation_mapping_migration', fields=['capture', *CAPTURE_ITEM_MAPPING])


__all__ = [
    'CAPTURE_IMAGE_MAPPING',
    'CAPTURE_ITEM_MAPPING',
    'LOCATION_FIELDS',
    'capture_image_block',
    'capture_item_fields',
    'ensure_capture_fields',
]
