"""One chunk of an assist run: read the items, ask the VLM, write what may be written.

The class answer goes through the shared class write path
(:mod:`~src.services.curation.vlm_class_batch`), so a human- or import-locked,
validated or holdout item never has its class touched; its answer is kept as a
suggestion in the item's ``assist`` slot like every other answer. Attributes are
only ever written to that slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.crops import mget_crops
from src.services.curation.assist.proposal import ClassAnswer
from src.services.curation.assist.slot_write import RunMeta, build_slot, write_slots
from src.services.curation.class_write_guard import CLASS_GUARD_SOURCE_FIELDS, class_write_locked
from src.services.curation.vlm_class_attempt import NEW_CLASS_SENTINEL, prediction_empty_reason
from src.services.curation.vlm_class_batch import (
    VlmContext,
    chunk_crops,
    updates_from_predictions,
    write_updates,
)
from src.services.labeling.detector_hint import DETECTOR_HINT_FIELDS


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.labeling.vlm_models import AttributeValue

WRITER = 'assist_run'


@dataclass
class ChunkOutcome:
    predicted: int = 0
    updated: int = 0
    failed: int = 0
    unreadable: int = 0
    locked_suggest_only: int = 0
    items_written: int = 0
    invalid_values: int = 0
    answers: list[ClassAnswer] = field(default_factory=list)
    verify_boxes: int = 0
    verify_accepted: int = 0
    verify_rejected: int = 0
    verify_locked: int = 0


def _suggested_name(update: dict[str, Any] | None, pred: Any) -> str | None:
    """The class the VLM named: the registry name when it resolved, else its own
    words (an unmatched label, or the slug it proposed)."""
    if update is not None and update.get('class_source') == 'vlm':
        return str(update['class_name'])
    raw = pred.proposed_class if pred.class_name == NEW_CLASS_SENTINEL else pred.class_name
    return raw.strip().lower()[:64] or None


def _answer(update: dict[str, Any] | None, pred: Any) -> ClassAnswer:
    """The vote one prediction casts: the registry class it resolved to, else the
    class it named, else (a `__new__` answer) the slug it proposed."""
    resolved = str(update['class_name']) if update and update.get('class_source') == 'vlm' else ''
    is_new = pred.class_name == NEW_CLASS_SENTINEL
    named = '' if is_new else pred.class_name.strip().lower()[:64]
    return ClassAnswer(
        class_name=resolved or named or None,
        proposed='' if resolved or named else pred.proposed_class,
        attributes={key: a.value for key, a in pred.attributes.items()},
    )


async def _classify(
    opensearch: AsyncOpenSearch,
    ctx: VlmContext,
    crops: list[Any],
    sources: dict[str, dict[str, Any]],
    *,
    with_attributes: bool,
    out: ChunkOutcome,
) -> tuple[dict[str, tuple[str | None, str | None]], dict[str, dict[str, AttributeValue]]]:
    """``({crop_id: (class name, confidence)}, {crop_id: attributes})`` of the
    usable answers; the class write itself is done here."""
    if with_attributes:
        preds = await ctx.labeler.assist_batch(
            crops, ctx.class_names, class_catalog=ctx.class_catalog
        )
    else:
        preds = await ctx.labeler.label_or_propose_batch(
            crops, ctx.class_names, class_catalog=ctx.class_catalog
        )
    out.predicted += len(preds)
    out.failed += sum(1 for p in preds if p.failure == 'request_failed')
    updates, _proposals = updates_from_predictions(preds, ctx, set_cluster=False)
    locked = {cid for cid in updates if class_write_locked(sources.get(cid, {}))}
    writable = {cid: u for cid, u in updates.items() if cid not in locked}
    await write_updates(opensearch, ctx, writable, writer_id=WRITER)
    out.updated += len(writable)
    classes: dict[str, tuple[str | None, str | None]] = {}
    attributes: dict[str, dict[str, AttributeValue]] = {}
    for p in preds:
        if p.failure is not None:
            continue
        answered = prediction_empty_reason(p) is None
        if answered and p.img_id in locked:
            out.locked_suggest_only += 1
        classes[p.img_id] = (
            _suggested_name(updates.get(p.img_id), p) if answered else None,
            p.confidence,
        )
        attributes[p.img_id] = p.attributes
        out.invalid_values += p.invalid_attribute_values
        if answered:
            out.answers.append(_answer(updates.get(p.img_id), p))
    return classes, attributes


async def process_chunk(
    opensearch: AsyncOpenSearch,
    ctx: VlmContext,
    ids: list[str],
    *,
    classify: bool,
    attributes: bool,
    meta: RunMeta,
) -> ChunkOutcome:
    out = ChunkOutcome()
    docs = await mget_crops(
        opensearch,
        ids,
        source_includes=[
            'image_path',
            'bbox_norm',
            'test_holdout',
            *DETECTOR_HINT_FIELDS,
            *CLASS_GUARD_SOURCE_FIELDS,
        ],
    )
    # Defense in depth: scope resolution already drops the frozen holdout.
    docs = {cid: d for cid, d in docs.items() if not (d.get('_source') or {}).get('test_holdout')}
    sources = {cid: d.get('_source') or {} for cid, d in docs.items()}
    for cid, source in sources.items():
        ctx.guard.remember(cid, source)
    crops = chunk_crops(docs, ctx)
    out.unreadable = len(docs) - len(crops)
    if not crops:
        return out
    classes: dict[str, tuple[str | None, str | None]] = {}
    found: dict[str, dict[str, AttributeValue]] = {}
    if classify:
        classes, found = await _classify(
            opensearch, ctx, crops, sources, with_attributes=attributes, out=out
        )
    else:
        preds = await ctx.labeler.attributes_batch(crops)
        out.predicted += len(preds)
        out.failed += sum(1 for p in preds if p.failure == 'request_failed')
        for p in preds:
            if p.failure is None:
                classes[p.img_id] = (None, None)
                found[p.img_id] = p.attributes
                out.invalid_values += p.invalid_values
                out.answers.append(
                    ClassAnswer(
                        class_name=None, attributes={k: a.value for k, a in p.attributes.items()}
                    )
                )
    slots = {
        cid: build_slot(
            meta,
            class_name=name if classify else None,
            class_confidence=conf if classify else None,
            attributes=found.get(cid, {}) if attributes else {},
        )
        for cid, (name, conf) in classes.items()
    }
    out.items_written = await write_slots(opensearch, slots)
    return out


__all__ = ['WRITER', 'ChunkOutcome', 'process_chunk']
