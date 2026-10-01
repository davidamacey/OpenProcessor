"""Per-box region-embedding stage for the curation detection worker.

Embeds (PE-Core, 1024-d, L2-normalized) every box a pass accepted -- the
live counterpart to ``scripts/curation/backfill_region_embeddings.py``,
which catches boxes that arrived without one (human-drawn boxes, items
accepted before this stage existed). Region clustering and
``GET /regions/suspected_false_positives``
(:mod:`src.services.curation.clustering.region_box_clustering`) key off
these vectors.

Shares the crop-to-vector step
(:func:`src.services.detection.region_embed.embed_region_crops`) with the
backfill script, so both write the exact same vector for the same crop.

Box ids are only final inside the write-time merge
(:func:`~scripts.curation.worker.bulk_writer._merge` mints them against the
live doc), so this stage keys its vectors by the box's pre-merge id
(``_ItemTask.box_vectors``); the merge maps them onto the final ids and
:func:`~scripts.curation.worker.bulk_writer._write_box_embeddings` writes
them after the box list landed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scripts.curation.worker.cascade import _crop_region_jpeg, _source_to_crop
from src.core.logging import get_logger
from src.services.detection.region_embed import embed_region_crops


if TYPE_CHECKING:
    from scripts.curation.worker.state import _ItemTask
    from src.clients.pe_encoder import PEEncoder
    from src.services.curation.region_boxes import RegionBox

logger = get_logger('curation_worker')

# Matches the backfill script's batch size (scripts/curation/backfill_region_embeddings.py).
ENCODE_BATCH = 32


def _accepted_boxes(task: _ItemTask) -> list[RegionBox]:
    """The boxes this pass resolved to ``accepted``, when the item crop is
    loaded (a rejected or proposed box is never embedded)."""
    if task.crop_jpeg is None or not task.pending_boxes:
        return []
    return [b for b in task.pending_boxes if b.state == 'accepted']


async def embed_written_regions(tasks: list[_ItemTask], pe: PEEncoder | Any) -> None:
    """Set ``task.box_vectors[box_id]`` for every accepted box of every task,
    in batches of :data:`ENCODE_BATCH`.

    Best-effort: an encoder failure for a batch is logged once and that
    batch is skipped -- those boxes carry no vector (the backfill script
    picks them up on its next run), but the flush this stage feeds into
    must never fail because of it.
    """
    pairs = [(t, b) for t in tasks for b in _accepted_boxes(t)]
    for start in range(0, len(pairs), ENCODE_BATCH):
        batch = pairs[start : start + ENCODE_BATCH]
        jpegs = [
            _crop_region_jpeg(t.crop_jpeg, _source_to_crop(b.bbox_norm, t.item_bbox_norm))  # type: ignore[arg-type]
            for t, b in batch
        ]
        try:
            embeddings = await embed_region_crops(pe, jpegs)
        except Exception as exc:
            logger.warning('region_embed_batch_failed', error=str(exc), batch_size=len(batch))
            continue
        for (t, b), emb in zip(batch, embeddings, strict=True):
            if emb is not None:
                t.box_vectors[b.box_id] = emb


__all__ = ['ENCODE_BATCH', 'embed_written_regions']
