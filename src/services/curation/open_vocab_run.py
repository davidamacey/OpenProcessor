"""The full-image SAM 3 pass: run an active prompt set on one stored image and
write every surviving hit as a normal item.

One function, :func:`run_open_vocab_image`, serves the reprocess scope and the
ingest-time hook. The segmenter call is injected (the API process and the
tests pass different callables); what it raises on an outage is
:class:`~src.services.detection.segmenter_http.SegmenterCallError`, which
aborts the image BEFORE anything is written or deleted: an outage is never "no
hit", and the image stays as it was.

Writing goes through the same :func:`~src.services.curation.ingest_index.index_items`
ingest uses, so a hit gets its crop, embedding, cluster and region seed like a
detector item. Ids are deterministic in (image, box), so a re-run upserts. Own
output of the SAME set that a re-run no longer produces is removed; a locked
item (human or imported label, locked box) is never overwritten, replaced or
deleted: a hit overlapping one is skipped and counted.
"""

from __future__ import annotations

import asyncio
import io
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from src.clients.curation_opensearch import get_class_registry
from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.curation.class_ensure import ensure_class_by_name
from src.services.curation.ingest_class_sources import OPEN_VOCAB_CLASS_SOURCE
from src.services.curation.ingest_index import index_items
from src.services.curation.item_delete import delete_items
from src.services.curation.item_doc import DetectedItem
from src.services.curation.reprocess_detect import load_image_context
from src.services.curation.reprocess_locks import item_locked
from src.services.curation.reprocess_targets import items_by_terms
from src.services.detection.geometry import crop_id, stored_bbox_norm
from src.services.detection.open_vocab_select import ExistingBox, Hit, select_open_vocab_hits


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch
    from PIL import Image

    from src.services.curation.ingest import CurationIngestService
    from src.services.detection.cascade_detect import RegionCandidate
    from src.services.detection.open_vocab_set import OpenVocabSet, OpenVocabTarget

logger = get_logger(__name__)

#: ``class_detector`` stamped on every hit (the segmenter, not a profile).
OPEN_VOCAB_DETECTOR = 'sam3'
OPEN_VOCAB_DETECTOR_VERSION = '1'
#: Registry group of classes created by a prompt target.
OPEN_VOCAB_CLASS_GROUP = 'open_vocab'
_JPEG_QUALITY = 90
_POLYGON_DECIMALS = 5


class SegmentImage(Protocol):
    async def __call__(
        self,
        jpeg: bytes,
        prompt: str,
        *,
        min_score: float | None,
        max_candidates: int,
        return_masks: bool,
    ) -> list[RegionCandidate]: ...


@dataclass
class OpenVocabImageResult:
    """What one image's pass did. ``dropped`` counts every discarded candidate
    by reason (the select reasons plus ``id_collision``)."""

    calls: int = 0
    hits: int = 0
    written: int = 0
    removed: int = 0
    locked_untouched: int = 0
    dropped: Counter[str] = field(default_factory=Counter)

    def as_counts(self) -> dict[str, int]:
        return {
            'calls': self.calls,
            'hits': self.hits,
            'written': self.written,
            'removed': self.removed,
            'locked_untouched': self.locked_untouched,
            **{f'dropped_{k}': v for k, v in sorted(self.dropped.items())},
        }


def encode_for_segmenter(pil: Image.Image, max_side: int) -> bytes:
    """JPEG of ``pil``, scaled down (never up) so its long side is at most
    ``max_side``. The frame is the whole image, so returned normalized
    coordinates need no projection: scaling preserves them."""
    img = pil.convert('RGB')
    long_side = max(img.size)
    if long_side > max_side:
        scale = max_side / long_side
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        )
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=_JPEG_QUALITY)
    return buf.getvalue()


def _existing_includes() -> list[str]:
    from src.config.region_fields import get_region_fields

    F = get_region_fields()
    return [
        'bbox_norm',
        'class_id',
        'class_name',
        'proposal_name',
        'class_source',
        'class_validated',
        'test_holdout',
        'open_vocab_set',
        F.boxes,
        F.validated,
        F.verifier,
    ]


def _polygon(candidate: RegionCandidate) -> list[list[float]] | None:
    if candidate.mask_polygon is None:
        return None
    return [
        [round(x, _POLYGON_DECIMALS), round(y, _POLYGON_DECIMALS)]
        for x, y in candidate.mask_polygon
    ]


def _detected(
    hit: Hit,
    target: OpenVocabTarget,
    ov: OpenVocabSet,
    revision: int | None,
    class_id: int | None,
    width: int,
    height: int,
) -> DetectedItem:
    x1, y1, x2, y2 = hit.candidate.bbox_norm
    extra: dict[str, Any] = {'source_prompt': hit.prompt, 'open_vocab_set': ov.name}
    if revision is not None:
        extra['open_vocab_revision'] = revision
    polygon = _polygon(hit.candidate) if target.mask else None
    if polygon is not None:
        extra['mask_polygon'] = polygon
    return DetectedItem(
        bbox_pixel=(x1 * width, y1 * height, x2 * width, y2 * height),
        score=hit.candidate.score,
        class_id=class_id,
        class_name=hit.class_name or None,
        class_source=OPEN_VOCAB_CLASS_SOURCE,
        # Discovery mode: no class yet, the prompt names the proposal.
        proposal_name=None if hit.class_name else hit.prompt,
        class_detector=OPEN_VOCAB_DETECTOR,
        class_detector_version=OPEN_VOCAB_DETECTOR_VERSION,
        extra_fields=extra,
    )


async def _segment_targets(
    segment: SegmentImage, jpeg: bytes, targets: tuple[OpenVocabTarget, ...]
) -> list[list[RegionCandidate]]:
    return list(
        await asyncio.gather(
            *(
                segment(
                    jpeg,
                    t.prompt,
                    min_score=t.min_score,
                    max_candidates=t.max_instances,
                    return_masks=t.mask,
                )
                for t in targets
            )
        )
    )


async def run_open_vocab_image(
    opensearch: AsyncOpenSearch,
    service: CurationIngestService,
    image_id: str,
    image_doc: dict[str, Any],
    ov: OpenVocabSet,
    *,
    revision: int | None,
    segment: SegmentImage,
    targets: tuple[OpenVocabTarget, ...] | None = None,
) -> OpenVocabImageResult:
    """Run ``targets`` (default: every enabled one) of ``ov`` on one stored
    image and write the surviving hits. Raises ``ValueError`` for an
    unservable or undecodable image and ``SegmenterCallError`` on an outage;
    in both cases nothing was written."""
    cfg = get_curation_config()
    ctx = await load_image_context(service, image_id, image_doc)
    todo = ov.enabled_targets if targets is None else targets
    result = OpenVocabImageResult()
    if not todo:
        return result

    jpeg = await asyncio.to_thread(encode_for_segmenter, ctx.pil, ov.image_max_side)
    candidates = await _segment_targets(segment, jpeg, todo)
    result.calls = len(todo)

    existing_docs = await items_by_terms(
        opensearch, 'image_id', [image_id], index=cfg.items_index, includes=_existing_includes()
    )
    existing: list[ExistingBox] = []
    own: set[str] = set()
    existing_ids: dict[str, str | None] = {}
    for doc_id, src in existing_docs:
        box = src.get('bbox_norm')
        if not box or len(box) != 4:
            continue
        locked = item_locked(src)
        if src.get('open_vocab_set') == ov.name and not locked:
            own.add(doc_id)
            continue
        existing_ids[doc_id] = src.get('class_name') or src.get('proposal_name')
        existing.append(
            ExistingBox(
                bbox_norm=(box[0], box[1], box[2], box[3]),
                class_name=src.get('class_name') or src.get('proposal_name'),
                locked=locked,
            )
        )
        result.locked_untouched += int(locked)

    selection = select_open_vocab_hits(
        [(t.rules(), c) for t, c in zip(todo, candidates, strict=True)],
        existing,
        dedup_iou=ov.dedup_iou,
    )
    result.dropped.update(reason for _hit, reason in selection.dropped)

    # An item's id is its (image, box): a hit on exactly the box of another
    # item would overwrite that item, so it is dropped before any write.
    seen: set[str] = set()
    kept: list[Hit] = []
    for h in selection.kept:
        cid = crop_id(image_id, stored_bbox_norm(h.candidate.bbox_norm, ctx.width, ctx.height))
        if cid in existing_ids or cid in seen:
            result.dropped['id_collision'] += 1
            continue
        seen.add(cid)
        kept.append(h)
    result.hits = len(kept)

    by_prompt = {t.prompt: t for t in todo}
    registry = get_class_registry()
    class_ids: dict[str, int] = {}
    for name in {h.class_name for h in kept if h.class_name}:
        class_ids[name] = ensure_class_by_name(
            registry, name, group=OPEN_VOCAB_CLASS_GROUP, notes='created by an open-vocabulary pass'
        )
    items = [
        _detected(
            h,
            by_prompt[h.prompt],
            ov,
            revision,
            class_ids.get(h.class_name),
            ctx.width,
            ctx.height,
        )
        for h in kept
    ]

    new_ids: list[str] = []
    if items:
        outcome = await index_items(service, ctx, items)
        if outcome.result.status != 'success':
            raise RuntimeError(outcome.result.error or 'index_items failed')
        new_ids = outcome.crop_ids
        result.written = len(new_ids)

    stale = sorted(own - set(new_ids))
    if stale:

        async def still_unlocked(_crop_id: str, doc: dict[str, Any]) -> bool:
            return not item_locked(doc)

        deleted = await delete_items(
            opensearch,
            stale,
            items_index=cfg.items_index,
            crop_cache_dir=cfg.crop_cache_dir,
            deletable=still_unlocked,
        )
        if deleted['errors']:
            raise RuntimeError(f'delete failed: {deleted["errors"][:3]}')
        result.removed = len(stale) - len(deleted['skipped'])
        result.locked_untouched += len(deleted['skipped'])
    return result


__all__ = [
    'OPEN_VOCAB_CLASS_GROUP',
    'OPEN_VOCAB_DETECTOR',
    'OPEN_VOCAB_DETECTOR_VERSION',
    'OpenVocabImageResult',
    'SegmentImage',
    'encode_for_segmenter',
    'run_open_vocab_image',
]
