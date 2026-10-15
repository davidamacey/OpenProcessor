"""The ``verify`` action: the VLM judges, box by box, whether a stored box is a real
region, and each verdict is written onto its own box.

Moved here from the retired ``/vlm/verify_regions`` route. One region crop is sent
per stored box still open to a machine verdict (``proposed`` or ``accepted``, never a
human- or import-owned one); a box the VLM gave no usable answer for is left untouched
for a retry, never written as rejected. The write is an OCC merge on the freshest
document, so a human verdict that landed during the VLM calls wins outright.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import Image

from src.clients.curation_opensearch.crops import mget_crops
from src.clients.occ import occ_skip_on_conflict_bulk
from src.clients.occ_locks import is_locked_box
from src.config.curation_accessors import items_index
from src.config.region_fields import get_region_fields
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.domain.region_boxes import read_boxes
from src.services.curation.assist.chunk import ChunkOutcome
from src.services.curation.crop_bytes import load_region_jpeg
from src.services.curation.region_verify import BoxVerdict, verifiable_boxes, verify_regions_update
from src.services.labeling.run_labeler import vlm_stamp
from src.services.labeling.vlm_models import RegionCrop
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Collection

    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

_READ_ERRORS = (RequestRejectedError, OSError, ValueError, Image.DecompressionBombError)


async def verify_chunk(
    opensearch: AsyncOpenSearch,
    labeler: Any,
    ids: list[str],
    *,
    box_ids: dict[str, list[str]] | None,
    pack_stamp: str,
) -> ChunkOutcome:
    """Verify the boxes of ``ids`` (only ``box_ids[crop_id]`` when given)."""
    F = get_region_fields()
    out = ChunkOutcome()
    docs = await mget_crops(
        opensearch,
        ids,
        index=items_index(),
        source_includes=[F.boxes, F.verifier, F.label_source, 'image_path'],
    )
    verdicts_by_id: dict[str, list[BoxVerdict]] = {}
    for crop_id in ids:
        src = (docs.get(crop_id) or {}).get('_source') or {}
        image_path = src.get('image_path', '')
        wanted: Collection[str] | None = None if box_ids is None else box_ids.get(crop_id, [])
        out.verify_locked += sum(
            1
            for b in read_boxes(src, F)
            if is_locked_box(b) and (wanted is None or b.box_id in wanted)
        )
        if not image_path:
            continue
        for box in verifiable_boxes(src, F, box_ids=wanted):
            try:
                jpeg = load_region_jpeg(image_path, tuple(box.bbox_norm))
            except _READ_ERRORS as exc:
                logger.warning(
                    'assist_verify_thumb_failed', crop_id=crop_id, box_id=box.box_id, error=str(exc)
                )
                out.unreadable += 1
                continue
            verdict = await labeler.verify_region(RegionCrop(crop_id=crop_id, jpeg_bytes=jpeg))
            if verdict is None:
                out.failed += 1
                continue
            out.verify_boxes += 1
            out.verify_accepted += int(bool(verdict.is_region))
            out.verify_rejected += int(not verdict.is_region)
            verdicts_by_id.setdefault(crop_id, []).append(
                BoxVerdict(
                    box_id=box.box_id,
                    bbox_norm=box.bbox_norm,
                    is_region=bool(verdict.is_region),
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                )
            )
    if verdicts_by_id:
        now = now_iso()

        def merge(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
            return verify_regions_update(
                current,
                verdicts_by_id[doc_id],
                now=now,
                pack_stamp=pack_stamp,
                vlm_stamp=vlm_stamp(labeler),
            )

        await occ_skip_on_conflict_bulk(
            opensearch,
            doc_ids=list(verdicts_by_id),
            merger=merge,
            index=items_index(),
            refresh=False,
            writer_id='assist_run',
        )
    return out


__all__ = ['verify_chunk']
