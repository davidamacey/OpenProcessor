"""Box attributes: the pack's `box` attributes reported for chosen boxes.

Each box is sent as its own close-up. The answers are written under
``assist.boxes[]`` (keyed by ``box_id``), never into the box itself, so the fixed
``RegionBox`` keys and the lock rule are untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import Image

from src.clients.curation_opensearch.crops import mget_crops
from src.config.curation_accessors import items_index
from src.config.region_fields import get_region_fields
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.services.curation.assist.chunk import ChunkOutcome
from src.services.curation.assist.slot_write import RunMeta, write_box_attributes
from src.services.curation.crop_bytes import load_region_jpeg
from src.services.labeling.vlm_models import ItemCrop


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

_READ_ERRORS = (RequestRejectedError, OSError, ValueError, Image.DecompressionBombError)
_SEP = '#'


async def box_attributes_chunk(
    opensearch: AsyncOpenSearch,
    labeler: Any,
    ids: list[str],
    *,
    box_ids: dict[str, list[str]],
    meta: RunMeta,
) -> ChunkOutcome:
    out = ChunkOutcome()
    F = get_region_fields()
    docs = await mget_crops(
        opensearch, ids, index=items_index(), source_includes=[F.boxes, 'image_path']
    )
    crops: list[ItemCrop] = []
    from src.domain.region_boxes import read_boxes

    for crop_id in ids:
        src = (docs.get(crop_id) or {}).get('_source') or {}
        stored = {b.box_id: b for b in read_boxes(src, F)}
        for box_id in box_ids.get(crop_id, []):
            box = stored.get(box_id)
            if box is None or not src.get('image_path'):
                continue
            try:
                jpeg = load_region_jpeg(src['image_path'], tuple(box.bbox_norm))
            except _READ_ERRORS as exc:
                logger.warning('assist_box_thumb_failed', crop_id=crop_id, error=str(exc))
                out.unreadable += 1
                continue
            crops.append(ItemCrop(img_id=f'{crop_id}{_SEP}{box_id}', jpeg_bytes=jpeg))
    if not crops:
        return out
    preds = await labeler.attributes_batch(crops, applies_to='box')
    out.predicted += len(preds)
    out.failed += sum(1 for p in preds if p.failure == 'request_failed')
    by_crop: dict[str, dict[str, list[dict[str, str]]]] = {}
    for p in preds:
        if p.failure is not None:
            continue
        crop_id, _, box_id = p.img_id.partition(_SEP)
        out.invalid_values += p.invalid_values
        by_crop.setdefault(crop_id, {})[box_id] = [
            {'key': k, 'value': a.value, 'confidence': a.confidence}
            for k, a in p.attributes.items()
        ]
    out.items_written = await write_box_attributes(opensearch, meta, by_crop)
    return out


__all__ = ['box_attributes_chunk']
