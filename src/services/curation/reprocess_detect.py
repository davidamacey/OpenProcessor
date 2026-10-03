"""The ``detect`` scope (W10.13): re-run the ingest detectors on a stored
image and merge the output into the image's items.

The merge is :func:`~src.services.curation.proposal_merge.merge_item_proposals`
with ``remove_stale``: a proposal that overlaps a locked item (human or
imported label, a locked box, a holdout item) only adds a ``proposal_chain``
note to it; an unlocked machine item the detector reproduces is refreshed in
place (same ``crop_id``); one whose box moved is replaced; an unmatched
proposal becomes a new machine item (region-seeded as ingest does); an
unlocked machine item nothing matched is deleted. A locked item is never
written, replaced or deleted.

The image bytes are read from the stored ``image_path`` only if it is a
servable path (:func:`~...image_serving.is_servable_image_path`); anything
else counts as a failed image, never a read.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_update_one
from src.config import get_curation_config
from src.config.region_fields import get_region_fields
from src.core.logging import get_logger
from src.services.curation.image_serving import is_servable_image_path
from src.services.curation.ingest_index import ImageContext, index_items
from src.services.curation.item_delete import delete_items
from src.services.curation.proposal_merge import (
    ExistingItem,
    MergePlan,
    Proposal,
    merge_item_proposals,
)
from src.services.curation.reprocess_locks import item_locked
from src.services.curation.reprocess_targets import items_by_terms
from src.services.detection.geometry import bbox_norm as _bbox_norm


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.item_doc import DetectedItem

logger = get_logger(__name__)

REPROCESS_SOURCE = 'reprocess'


def _existing_includes() -> list[str]:
    F = get_region_fields()
    return [
        'image_id',
        'bbox_norm',
        'class_id',
        'class_name',
        'class_source',
        'class_validated',
        'test_holdout',
        'proposal_chain',
        F.boxes,
        F.validated,
        F.verifier,
    ]


def _proposal(item: DetectedItem, width: int, height: int) -> Proposal:
    bn = _bbox_norm(item.bbox_pixel, width, height)
    return Proposal(
        bbox_norm=(bn[0], bn[1], bn[2], bn[3]),
        class_id=item.class_id,
        class_name=item.class_name,
        score=float(item.score),
    )


def _pixel_box(
    bbox_norm: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    return (
        bbox_norm[0] * width,
        bbox_norm[1] * height,
        bbox_norm[2] * width,
        bbox_norm[3] * height,
    )


def _note_chain(note: str):
    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        chain = list(current.get('proposal_chain') or [])
        if note in chain:
            return {}
        return {'proposal_chain': [*chain, note]}

    return _merge


async def _apply_plan(
    opensearch: AsyncOpenSearch,
    service: CurationIngestService,
    ctx: ImageContext,
    plan: MergePlan,
    by_proposal: dict[Proposal, DetectedItem],
    existing_bbox: dict[str, tuple[float, float, float, float]],
) -> list[str]:
    """Apply ``plan``. Returns the stale ids left in place because a human
    locked them while the detector ran."""
    cfg = get_curation_config()
    note = f'{service.profile.name}:match'
    for crop_id, _prop in plan.merged_into_locked:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=_note_chain(note),
            index=cfg.items_index,
            writer_id='reprocess_detect',
        )

    to_index: list[DetectedItem] = []
    for crop_id, prop in plan.refreshed:
        item = by_proposal[prop]
        # Keep the stored box so the refreshed doc keeps its crop_id.
        item.bbox_pixel = _pixel_box(existing_bbox[crop_id], ctx.width, ctx.height)
        to_index.append(item)
    for _old, prop in plan.replaced:
        to_index.append(by_proposal[prop])
    to_index.extend(by_proposal[prop] for prop in plan.created)
    if to_index:
        outcome = await index_items(service, ctx, to_index)
        if outcome.result.status != 'success':
            raise RuntimeError(outcome.result.error or 'index_items failed')
    stale = [cid for cid, _ in plan.replaced] + list(plan.removed)
    if not stale:
        return []

    async def still_unlocked(_crop_id: str, doc: dict[str, Any]) -> bool:
        return not item_locked(doc)

    result = await delete_items(
        opensearch,
        stale,
        items_index=cfg.items_index,
        crop_cache_dir=cfg.crop_cache_dir,
        deletable=still_unlocked,
    )
    if result['errors']:
        raise RuntimeError(f'delete failed: {result["errors"][:3]}')
    return list(result['skipped'])


async def redetect_image(
    opensearch: AsyncOpenSearch,
    service: CurationIngestService,
    image_id: str,
    image_doc: dict[str, Any],
) -> dict[str, int]:
    """Re-detect one stored image. Returns the merge counters
    (``merged``/``refreshed``/``replaced``/``created``/``removed``/
    ``locked_untouched``); raises on an unservable or unreadable image."""
    cfg = get_curation_config()
    path = image_doc.get('image_path') or ''
    if not is_servable_image_path(path):
        raise ValueError(f'image path is not under a configured source root: {path!r}')
    data = await asyncio.to_thread(Path(path).read_bytes)
    ctx = await service.ingest_image(data, path, REPROCESS_SOURCE, adopt_existing=True)
    if not isinstance(ctx, ImageContext):
        raise ValueError(ctx.error or 'image could not be decoded')
    # Dedup may resolve the bytes to a different image doc with the same
    # content; this reprocess is for the requested one.
    ctx.image_id = image_id
    ctx.image_path = path
    ctx.created = False

    existing_docs = await items_by_terms(
        opensearch, 'image_id', [image_id], index=cfg.items_index, includes=_existing_includes()
    )
    existing: list[ExistingItem] = []
    for crop_id, src in existing_docs:
        bn = src.get('bbox_norm')
        if not bn or len(bn) != 4:
            continue
        existing.append(
            ExistingItem(
                crop_id=crop_id,
                bbox_norm=(bn[0], bn[1], bn[2], bn[3]),
                class_name=src.get('class_name'),
                locked=item_locked(src),
            )
        )
    items = (await service.detect_items(ctx.pil, image_path=ctx.image_path)).items
    proposals = [_proposal(it, ctx.width, ctx.height) for it in items]
    by_proposal = dict(zip(proposals, items, strict=True))
    plan = merge_item_proposals(existing, proposals, remove_stale=True)
    existing_bbox = {e.crop_id: e.bbox_norm for e in existing}
    kept = await _apply_plan(opensearch, service, ctx, plan, by_proposal, existing_bbox)
    kept_replaced = len(set(kept) & {cid for cid, _ in plan.replaced})
    return {
        'merged': len(plan.merged_into_locked),
        'refreshed': len(plan.refreshed),
        'replaced': len(plan.replaced) - kept_replaced,
        'created': len(plan.created),
        'removed': len(plan.removed) - (len(kept) - kept_replaced),
        'locked_untouched': sum(1 for e in existing if e.locked) + len(kept),
    }


__all__ = ['REPROCESS_SOURCE', 'redetect_image']
