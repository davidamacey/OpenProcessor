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
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from src.clients.curation_opensearch import get_class_registry
from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.curation.class_ensure import ResolvedClass, ensure_class_by_name
from src.services.curation.ingest_class_sources import (
    OPEN_VOCAB_CLASS_SOURCE,
    OPEN_VOCAB_TARGET_CLASS_SOURCE,
)
from src.services.curation.ingest_index import index_items
from src.services.curation.item_delete import delete_items
from src.services.curation.item_doc import DetectedItem
from src.services.curation.metrics import (
    OP_OPEN_VOCAB_CALL_SECONDS,
    OP_OPEN_VOCAB_HITS_DROPPED_TOTAL,
    OP_OPEN_VOCAB_ITEMS_WRITTEN_TOTAL,
    OP_SEGMENTER_GATE_DECISIONS_TOTAL,
)
from src.services.curation.open_vocab_gate import VlmVisibleFn  # noqa: TC001 - dataclass field type
from src.services.curation.reprocess_detect import load_image_context
from src.services.curation.reprocess_locks import item_locked
from src.services.curation.reprocess_targets import items_by_terms
from src.services.detection import segmenter_latency
from src.services.detection.geometry import crop_id, stored_bbox_norm
from src.services.detection.open_vocab_select import (
    ExistingBox,
    Hit,
    OpenVocabSelection,
    select_open_vocab_hits,
)
from src.services.detection.segmenter_gate import GateDecision, GateSubject, HitRateTracker, decide
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch
    from PIL import Image

    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.open_vocab_fields import OpenVocabStatus
    from src.services.detection.cascade_detect.candidate import RegionCandidate
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
    sampled: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    #: Targets the gate did not spend a call on, by ``tier<N>_<reason>``.
    skipped: Counter[str] = field(default_factory=Counter)

    def as_counts(self) -> dict[str, int]:
        return {
            'calls': self.calls,
            'hits': self.hits,
            'written': self.written,
            'removed': self.removed,
            'locked_untouched': self.locked_untouched,
            'gate_sampled': self.sampled,
            **{f'dropped_{k}': v for k, v in sorted(self.dropped.items())},
            **{f'skipped_gate_{k}': v for k, v in sorted(self.skipped.items())},
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
    registry_class: ResolvedClass | None,
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
        class_id=registry_class.class_id if registry_class else None,
        class_name=registry_class.class_name if registry_class else None,
        class_source=OPEN_VOCAB_TARGET_CLASS_SOURCE if registry_class else OPEN_VOCAB_CLASS_SOURCE,
        # Discovery mode: no class yet, the prompt names the proposal.
        proposal_name=None if hit.class_name else hit.prompt,
        class_detector=OPEN_VOCAB_DETECTOR,
        class_detector_version=OPEN_VOCAB_DETECTOR_VERSION,
        extra_fields=extra,
    )


async def _segment_target(
    segment: SegmentImage, jpeg: bytes, target: OpenVocabTarget
) -> list[RegionCandidate]:
    started = time.monotonic()
    outcome = 'error'
    try:
        found = await segment(
            jpeg,
            target.prompt,
            min_score=target.min_score,
            max_candidates=target.max_instances,
            return_masks=target.mask,
        )
        outcome = 'hit' if found else 'miss'
    finally:
        elapsed = time.monotonic() - started
        OP_OPEN_VOCAB_CALL_SECONDS.labels(outcome=outcome).observe(elapsed)
        if outcome != 'error':
            segmenter_latency.observe(elapsed)
    return found


async def _segment_targets(
    segment: SegmentImage, jpeg: bytes, targets: tuple[OpenVocabTarget, ...]
) -> list[list[RegionCandidate]]:
    return list(await asyncio.gather(*(_segment_target(segment, jpeg, t) for t in targets)))


def _count_decision(decision: GateDecision) -> None:
    kind = 'sample' if decision.sampled else 'run' if decision.run else 'skip'
    OP_SEGMENTER_GATE_DECISIONS_TOTAL.labels(
        scope='image',
        decision=kind,
        tier=str(decision.tier or ''),
        reason=decision.reason or '',
    ).inc()


@dataclass
class GateContext:
    """What the segmenter gate may consult beyond the registry rules: the
    vision model (tier 2) and the hit-rate windows (tier 3, updated in place).
    Either may be absent: that tier then does not run."""

    vlm_visible: VlmVisibleFn | None = None
    tracker: HitRateTracker | None = None


@dataclass
class ImagePlan:
    """The read-only half of a pass on one image: which targets the gate let
    through, what the segmenter said and what survives selection and dedup.
    Shared by the writing runner and the per-image test route, so a test shows
    exactly what a run would keep."""

    selection: OpenVocabSelection
    ran: tuple[OpenVocabTarget, ...]
    skipped: list[tuple[OpenVocabTarget, GateDecision]]
    #: Targets that ran only as a tier-3 recovery sample.
    sampled: int
    #: Own items of this set already on the image: ``{crop_id: source_prompt}``.
    own_ids: dict[str, str | None]
    existing_ids: dict[str, str | None]
    locked_untouched: int


async def plan_open_vocab_image(
    opensearch: AsyncOpenSearch,
    image_id: str,
    pil: Image.Image,
    ov: OpenVocabSet,
    targets: tuple[OpenVocabTarget, ...],
    *,
    segment: SegmentImage,
    gate: GateContext | None = None,
) -> ImagePlan:
    """Gate, segment and select for ``targets`` on ``pil`` against the image's
    items. Writes nothing. Raises ``SegmenterCallError`` on an outage."""
    gate = gate or GateContext()
    jpeg = await asyncio.to_thread(encode_for_segmenter, pil, ov.image_max_side)
    existing_docs = await items_by_terms(
        opensearch,
        'image_id',
        [image_id],
        index=get_curation_config().items_index,
        includes=_existing_includes(),
    )
    existing: list[ExistingBox] = []
    own: dict[str, str | None] = {}
    existing_ids: dict[str, str | None] = {}
    names: list[str | None] = []
    locked_count = 0
    for doc_id, src in existing_docs:
        box = src.get('bbox_norm')
        if not box or len(box) != 4:
            continue
        name = src.get('class_name') or src.get('proposal_name')
        names.append(name)
        locked = item_locked(src)
        if src.get('open_vocab_set') == ov.name and not locked:
            own[doc_id] = src.get('source_prompt')
            continue
        existing_ids[doc_id] = name
        existing.append(
            ExistingBox(bbox_norm=(box[0], box[1], box[2], box[3]), class_name=name, locked=locked)
        )
        locked_count += int(locked)

    subject = GateSubject(names=tuple(names))
    precheck = ov.gating.tier2_vlm_precheck and gate.vlm_visible is not None
    hit_cfg = ov.gating.tier3_hit_rate

    async def decision_for(t: OpenVocabTarget) -> GateDecision:
        ask = gate.vlm_visible
        return await decide(
            enabled=t.enabled,
            parent_classes=t.parent_classes,
            subject=subject,
            vlm_visible=(lambda: ask(jpeg, t.prompt)) if precheck and ask is not None else None,
            hit_rate=(gate.tracker, t.key, hit_cfg) if gate.tracker is not None else None,
        )

    decisions = await asyncio.gather(*(decision_for(t) for t in targets))
    for decision in decisions:
        _count_decision(decision)
    ran = tuple(t for t, d in zip(targets, decisions, strict=True) if d.run)
    skipped = [(t, d) for t, d in zip(targets, decisions, strict=True) if not d.run]

    candidates = await _segment_targets(segment, jpeg, ran)
    if gate.tracker is not None:
        for t, cands in zip(ran, candidates, strict=True):
            gate.tracker.record(t.key, hit=bool(cands), window=hit_cfg.window)
    selection = select_open_vocab_hits(
        [(t.rules(), c) for t, c in zip(ran, candidates, strict=True)],
        existing,
        dedup_iou=ov.dedup_iou,
    )
    return ImagePlan(
        selection,
        ran,
        skipped,
        sum(1 for d in decisions if d.run and d.sampled),
        own,
        existing_ids,
        locked_count,
    )


async def stamp_open_vocab_status(
    opensearch: AsyncOpenSearch, image_id: str, status: OpenVocabStatus
) -> None:
    """Record the pass's state on the image doc (see :data:`OpenVocabStatus`).
    The state is bookkeeping: a failed stamp is logged, never allowed to fail the image whose items are already written."""
    try:
        await opensearch.update(
            index=get_curation_config().images_index,
            id=image_id,
            body={
                'doc': {
                    'open_vocab_status': status,
                    'open_vocab_status_at': datetime.now(UTC).isoformat(),
                }
            },
        )
    except Exception as exc:
        logger.warning(
            'open_vocab_status_stamp_failed', image_id=image_id, status=status, error=str(exc)
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
    gate: GateContext | None = None,
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

    plan = await plan_open_vocab_image(
        opensearch, image_id, ctx.pil, ov, todo, segment=segment, gate=gate
    )
    selection, own, existing_ids = plan.selection, plan.own_ids, plan.existing_ids
    result.calls = len(plan.ran)
    result.locked_untouched = plan.locked_untouched
    result.dropped.update(reason for _hit, reason in selection.dropped)
    result.skipped.update(d.label for _t, d in plan.skipped)
    result.sampled = plan.sampled
    if not plan.ran:
        # Nothing was asked, so nothing is known: never write or remove.
        return result

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
    for reason, count in result.dropped.items():
        OP_OPEN_VOCAB_HITS_DROPPED_TOTAL.labels(reason=reason).inc(count)

    by_target = {(t.prompt, t.class_name): t for t in todo}
    registry = get_class_registry()
    resolved: dict[str, ResolvedClass] = {}
    for name in {h.class_name for h in kept if h.class_name}:
        resolved[name] = ensure_class_by_name(
            registry,
            normalize_class_name(name),
            group=OPEN_VOCAB_CLASS_GROUP,
            notes='created by an open-vocabulary pass',
        )
    items = [
        _detected(
            h,
            by_target[(h.prompt, h.class_name)],
            ov,
            revision,
            resolved.get(h.class_name),
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
        OP_OPEN_VOCAB_ITEMS_WRITTEN_TOTAL.inc(result.written)

    # A target the gate skipped was not looked at, so its earlier output stays.
    protected = {t.prompt for t, _d in plan.skipped}
    stale = sorted(i for i, prompt in own.items() if i not in new_ids and prompt not in protected)
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
    'GateContext',
    'ImagePlan',
    'OpenVocabImageResult',
    'SegmentImage',
    'encode_for_segmenter',
    'plan_open_vocab_image',
    'run_open_vocab_image',
    'stamp_open_vocab_status',
]
