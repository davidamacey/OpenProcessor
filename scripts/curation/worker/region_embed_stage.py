"""Per-box region-embedding stage for the curation detection worker.

Embeds (PE-Core, 1024-d, L2-normalized) every box a pass accepted -- the
live counterpart to ``scripts/curation/backfill_region_embeddings.py``,
which catches boxes that arrived without one (human-drawn boxes, items
accepted before this stage existed). Region clustering and
``GET /regions/suspected_false_positives``
(:mod:`src.services.curation.clustering.region_box_clustering`) key off
these vectors.

The crop of a box is its stored pixels cut from the decoded source image, the cut the
``embed`` reprocess scope and the GPU route make (``reprocess_embed.crop_arrays``), never
the item-crop JPEG: a JPEG round trip moves the vector of a small box (#360). The source
is decoded once per item for all of its boxes.

Box ids are only final inside the write-time merge
(:func:`~scripts.curation.worker.bulk_writer._merge` mints them against the
live doc), so this stage keys its vectors by the box's pre-merge id
(``_ItemTask.box_vectors``); the merge maps them onto the final ids and
:func:`~scripts.curation.worker.bulk_writer._write_box_embeddings` writes
them after the box list landed.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.logging import get_logger
from src.services.curation.crop_bytes import decode_source_rgb
from src.services.curation.reprocess_embed import crop_arrays
from src.utils.cpu_pool import run_cpu


if TYPE_CHECKING:
    from scripts.curation.worker.state import _ItemTask
    from src.clients.pe_encoder import PEEncoder
    from src.domain.region_boxes import RegionBox

logger = get_logger('curation_worker')

# Matches the backfill script's batch size (scripts/curation/backfill_region_embeddings.py).
ENCODE_BATCH = 32


def _accepted_boxes(task: _ItemTask) -> list[RegionBox]:
    """The boxes this pass resolved to ``accepted``, when the item crop is
    loaded (a rejected or proposed box is never embedded)."""
    if task.crop_jpeg is None or not task.pending_boxes:
        return []
    return [b for b in task.pending_boxes if b.state == 'accepted']


async def _task_crops(task: _ItemTask) -> list[tuple[RegionBox, np.ndarray]]:
    """``(box, pixels)`` of every accepted box of ``task``; empty when the source cannot be
    decoded (those boxes carry no vector; the backfill picks them up)."""
    boxes = _accepted_boxes(task)
    if not boxes:
        return []
    img = await asyncio.to_thread(decode_source_rgb, task.image_path)
    if img is None:
        return []
    arrays = await run_cpu(crop_arrays, img, [b.bbox_norm for b in boxes])
    return list(zip(boxes, arrays, strict=True))


async def embed_written_regions(tasks: list[_ItemTask], pe: PEEncoder | Any) -> None:
    """Set ``task.box_vectors[box_id]`` for every accepted box of every task,
    in batches of :data:`ENCODE_BATCH`.

    Best-effort: an encoder failure for a batch is logged once and that
    batch is skipped -- those boxes carry no vector (the backfill script
    picks them up on its next run), but the flush this stage feeds into
    must never fail because of it.
    """
    pairs = [(t, b, arr) for t in tasks for b, arr in await _task_crops(t)]
    for start in range(0, len(pairs), ENCODE_BATCH):
        batch = pairs[start : start + ENCODE_BATCH]
        try:
            embeddings = await pe.embed_crops([arr for _, _, arr in batch])
        except Exception as exc:
            logger.warning('region_embed_batch_failed', error=str(exc), batch_size=len(batch))
            continue
        for (t, b, _), emb in zip(batch, embeddings, strict=True):
            t.box_vectors[b.box_id] = np.asarray(emb, dtype=float).tolist()


__all__ = ['ENCODE_BATCH', 'embed_written_regions']
