"""Region-embedding write stage for the curation detection worker.

Sets ``RegionFields.embedding`` (PE-Core, 1024-d, L2-normalized) on every
task this pass accepted as ``detected`` -- the live counterpart to
``scripts/curation/backfill_region_embeddings.py``, which only catches
items ingested before this stage existed. Region-FP clustering
(:mod:`src.services.curation.clustering.orchestrator`) and
``GET /regions/suspected_false_positives`` both key off this field.

Shares the crop-to-vector step
(:func:`src.services.detection.region_embed.embed_region_crops`) with the
backfill script, so both write the exact same vector for the same crop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scripts.curation.worker.cascade import _crop_region_jpeg
from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.detection.region_embed import embed_region_crops


if TYPE_CHECKING:
    from scripts.curation.worker.state import _ItemTask
    from src.clients.pe_encoder import PEEncoder

logger = get_logger('curation_worker')

# Matches the backfill script's batch size (scripts/curation/backfill_region_embeddings.py).
ENCODE_BATCH = 32


def _eligible_tasks(tasks: list[_ItemTask]) -> list[_ItemTask]:
    """Tasks with a fresh ``detected`` write and an in-crop box to embed.

    ``candidate_in_crop`` is the accepted region's crop-relative box --
    populated for every acceptance path (primary detector, segmenter,
    text-hint re-pass) by write time; see
    ``scripts/curation/worker/region_text_stage.py`` for the same
    precondition used to OCR the region.
    """
    F = get_region_fields()
    return [
        t
        for t in tasks
        if t.update_doc.get(F.status) == RegionStatus.DETECTED
        and t.crop_jpeg is not None
        and t.candidate_in_crop is not None
    ]


async def embed_written_regions(tasks: list[_ItemTask], pe: PEEncoder | Any) -> None:
    """Set ``F.embedding`` on every eligible task's ``update_doc``, in
    batches of :data:`ENCODE_BATCH`.

    Best-effort: an encoder failure for a batch is logged once and that
    batch is skipped -- the field stays absent (the backfill script picks
    it up on its next run), but the flush this stage feeds into must never
    fail because of it.
    """
    F = get_region_fields()
    eligible = _eligible_tasks(tasks)
    if not eligible:
        return
    for start in range(0, len(eligible), ENCODE_BATCH):
        batch = eligible[start : start + ENCODE_BATCH]
        jpegs = [_crop_region_jpeg(t.crop_jpeg, t.candidate_in_crop) for t in batch]  # type: ignore[arg-type]
        try:
            embeddings = await embed_region_crops(pe, jpegs)
        except Exception as exc:
            logger.warning('region_embed_batch_failed', error=str(exc), batch_size=len(batch))
            continue
        for t, emb in zip(batch, embeddings, strict=True):
            if emb is not None:
                t.update_doc[F.embedding] = emb


__all__ = ['ENCODE_BATCH', 'embed_written_regions']
