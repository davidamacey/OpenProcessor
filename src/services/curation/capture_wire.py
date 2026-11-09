"""The `capture` object of the item wire, and the one place a coordinate could be added to it.

``capture_to_wire`` is a whitelist over a stored images-index `capture` block (or the two copies on
an item): the fields below go out, nothing else does, so `capture.location` and the other position
fields can never reach an item response by a new key being added to the stored block. The
coordinates are served by one route only (``GET /curation/projects/{project}/map/...``), which
reads them through ``apply_location_policy``.

``taken_local`` is a wall-clock time without a zone and ``taken_utc`` is set only when the offset
is known; ``has_location`` says a GPS fix was in the file (whatever the policy kept).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from opensearchpy.exceptions import NotFoundError

from src.config.curation_accessors import images_index


if TYPE_CHECKING:
    from collections.abc import Mapping

CAPTURE_WIRE_FIELDS: Final[tuple[str, ...]] = (
    'taken_local',
    'taken_utc',
    'offset_minutes',
    'time_source',
    'offset_source',
    'time_precision',
    'camera_make',
    'camera_model',
    'lens_model',
    'focal_length_mm',
    'focal_length_35mm',
    'orientation',
    'has_location',
    'exif_status',
)


def capture_to_wire(block: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The wire `capture` of a stored images-index block, or ``None`` without one."""
    if not block:
        return None
    return {name: block.get(name) for name in CAPTURE_WIRE_FIELDS}


def item_capture_to_wire(item: Mapping[str, Any]) -> dict[str, Any] | None:
    """The wire `capture` of an item document: the two capture times it carries, the rest null."""
    local, utc = item.get('capture_taken_local'), item.get('capture_taken_utc')
    if local is None and utc is None:
        return None
    return capture_to_wire({'taken_local': local, 'taken_utc': utc})


async def image_capture(opensearch: Any, image_id: str) -> dict[str, Any] | None:
    """The full wire `capture` of the image ``image_id``; ``None`` when the image has no capture
    block or is gone (the item then keeps the times it carries)."""
    if not image_id:
        return None
    try:
        resp = await opensearch.get(index=images_index(), id=image_id, _source_includes=['capture'])
    except NotFoundError:
        return None
    return capture_to_wire((resp.get('_source') or {}).get('capture'))


__all__ = ['CAPTURE_WIRE_FIELDS', 'capture_to_wire', 'image_capture', 'item_capture_to_wire']
