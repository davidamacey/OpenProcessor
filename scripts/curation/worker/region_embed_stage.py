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
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.logging import get_logger
from src.services.curation.crop_bytes import decode_source_rgb
from src.services.curation.reprocess_embed import crop_arrays
from src.utils.cpu_pool import inflight_mb, run_cpu


if TYPE_CHECKING:
    from scripts.curation.worker.state import _ItemTask
    from src.clients.pe_encoder import PEEncoder
    from src.domain.region_boxes import RegionBox

logger = get_logger('curation_worker')

# Matches the backfill script's batch size (scripts/curation/backfill_region_embeddings.py).
ENCODE_BATCH = 32

# RGB bytes of a 21 MP photo, in MiB: the memory one live decode is budgeted at.
DECODED_MB_21MP = 64


def decode_concurrency() -> int:
    """Photos decoded at once: half the CPUs, and no more than ``OP_INGEST_INFLIGHT_MB`` of
    21 MP frames, at least one."""
    by_cpu = (os.cpu_count() or 1) // 2
    by_memory = inflight_mb() // DECODED_MB_21MP
    return max(1, min(by_cpu, by_memory))


def _accepted_boxes(task: _ItemTask) -> list[RegionBox]:
    """The boxes this pass resolved to ``accepted``, when the item crop is
    loaded (a rejected or proposed box is never embedded)."""
    if task.crop_jpeg is None or not task.pending_boxes:
        return []
    return [b for b in task.pending_boxes if b.state == 'accepted']


async def _photo_crops(
    image_path: str, boxes: list[RegionBox], slots: asyncio.Semaphore
) -> list[np.ndarray] | None:
    """The pixels of ``boxes`` cut from one decode of ``image_path``; ``None`` when the source
    cannot be decoded (those boxes carry no vector; the backfill picks them up). The slot is
    held while the decoded frame is alive."""
    async with slots:
        img = await asyncio.to_thread(decode_source_rgb, image_path)
        if img is None:
            return None
        return await run_cpu(crop_arrays, img, [b.bbox_norm for b in boxes])


async def _task_crops(tasks: list[_ItemTask]) -> list[tuple[_ItemTask, RegionBox, np.ndarray]]:
    """``(task, box, pixels)`` of every accepted box, in task order, one decode per source."""
    by_path: dict[str, list[tuple[_ItemTask, RegionBox]]] = defaultdict(list)
    for t in tasks:
        for b in _accepted_boxes(t):
            by_path[t.image_path].append((t, b))
    slots = asyncio.Semaphore(decode_concurrency())
    paths = list(by_path)
    cut = await asyncio.gather(*(_photo_crops(p, [b for _, b in by_path[p]], slots) for p in paths))
    arrays: dict[tuple[int, str], np.ndarray] = {}
    for path, arrs in zip(paths, cut, strict=True):
        if arrs is not None:
            for (t, b), arr in zip(by_path[path], arrs, strict=True):
                arrays[id(t), b.box_id] = arr
    return [
        (t, b, arrays[id(t), b.box_id])
        for t in tasks
        for b in _accepted_boxes(t)
        if (id(t), b.box_id) in arrays
    ]


async def embed_written_regions(tasks: list[_ItemTask], pe: PEEncoder | Any) -> None:
    """Set ``task.box_vectors[box_id]`` for every accepted box of every task,
    in batches of :data:`ENCODE_BATCH`.

    Best-effort: an encoder failure for a batch is logged once and that
    batch is skipped -- those boxes carry no vector (the backfill script
    picks them up on its next run), but the flush this stage feeds into
    must never fail because of it.
    """
    pairs = await _task_crops(tasks)
    for start in range(0, len(pairs), ENCODE_BATCH):
        batch = pairs[start : start + ENCODE_BATCH]
        try:
            embeddings = await pe.embed_crops([arr for _, _, arr in batch])
        except Exception as exc:
            logger.warning('region_embed_batch_failed', error=str(exc), batch_size=len(batch))
            continue
        for (t, b, _), emb in zip(batch, embeddings, strict=True):
            t.box_vectors[b.box_id] = np.asarray(emb, dtype=float).tolist()


__all__ = ['DECODED_MB_21MP', 'ENCODE_BATCH', 'decode_concurrency', 'embed_written_regions']
