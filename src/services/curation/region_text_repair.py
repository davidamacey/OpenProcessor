"""Re-derive stored region text under the region-text validity rules.

A box can carry a VLM reading that is not text as its chosen ``text`` --
the prompt's example value ("ABC123"), a "can't read it" word, a stock run
("999") -- when the profile's text rules changed after it was read, even
where the OCR reader stored a valid reading. This re-runs the chooser
(:func:`~src.services.detection.region_text.resolve_region_text`) on each
box's *stored* readings -- ``text_vlm`` (or a VLM-sourced ``text``) and
``text_ocr`` -- with the deployment's current rules, and rewrites the
chosen-text attributes (:data:`MANAGED_ATTRS`) of that box. The per-reader
readings and ``text_raw`` are never changed, so the repair is re-runnable.

Human-typed text (``text_source='human'``) is never touched. An OCR
reading's own confidence isn't stored separately, so an OCR reading newly
chosen here gets ``text_confidence=null``; a choice that doesn't change
keeps its stored confidence and engine.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_skip_on_conflict_bulk
from src.config import get_region_fields
from src.services.curation.region_boxes import RegionBox, box_query, boxes_write_fields, read_boxes
from src.services.detection.region_text import (
    TEXT_SOURCE_OCR,
    TEXT_SOURCE_VLM,
    DominantTextConfig,
    DominantTextReading,
    ocr_engine_id,
    resolve_region_text,
)


if TYPE_CHECKING:
    from src.config import DetectionProfile
    from src.services.detection.region_text_rules import RegionTextRules


REPAIR_WRITER = 'operator:rederive_region_text'
HUMAN_TEXT_SOURCE = 'human'

# RegionBox attributes of the chosen reading, rewritten by the repair.
MANAGED_ATTRS: tuple[str, ...] = (
    'text',
    'text_source',
    'text_confidence',
    'text_engine_version',
    'text_disagreement',
    'text_choice',
    'text_vlm_invalid',
)


def _is_vlm_source(source: Any) -> bool:
    # Older writers stamped the VLM model id as the source.
    return bool(source) and source not in (TEXT_SOURCE_OCR, HUMAN_TEXT_SOURCE)


def _view(box: RegionBox) -> dict[str, Any]:
    """The text attributes of ``box`` (plus its raw reading) by bare name."""
    return {a: getattr(box, a) for a in (*MANAGED_ATTRS, 'text_vlm', 'text_ocr', 'text_raw')}


def stored_vlm_reading(view: dict[str, Any]) -> str | None:
    """The VLM's stored reading: ``text_vlm``, else a VLM-sourced ``text``."""
    vlm = view.get('text_vlm')
    if not vlm and _is_vlm_source(view.get('text_source')):
        vlm = view.get('text')
    return str(vlm) if vlm else None


def rederive(
    box: RegionBox,
    *,
    profile: DetectionProfile,
    rules: RegionTextRules,
    vlm_model: str | None = None,
) -> dict[str, Any] | None:
    """Target values (by attribute name) of :data:`MANAGED_ATTRS` for
    ``box``, or ``None`` when it holds human text or no stored reading at
    all, or the profile does not read text (a text-free profile has no
    chooser to re-run, and stored text is left as it is)."""
    if not profile.reads_text:
        return None
    view = _view(box)
    stored_source = view['text_source']
    if stored_source == HUMAN_TEXT_SOURCE:
        return None
    vlm = stored_vlm_reading(view)
    ocr_text = view['text_ocr']
    if not vlm and not ocr_text:
        return None
    ocr = (
        DominantTextReading(str(ocr_text), str(view['text_raw'] or ''), None, None, (), 'ok')
        if ocr_text
        else None
    )
    out = resolve_region_text(
        profile.text_reader,
        vlm_text=vlm,
        vlm_confidence=None,
        # The repair re-derives from readings already stored: the engine of
        # a VLM reading is the model that made it (stamped on the item at
        # write time; ``vlm_model``), not whatever this process is configured with.
        vlm_engine=vlm_model or '',
        ocr=ocr,
        ocr_engine=ocr_engine_id(profile),
        normalizer=DominantTextConfig.from_profile(profile).normalizer,
        rules=rules,
    )
    target = {a: out.get(a) for a in MANAGED_ATTRS}
    chosen = out.get('text_source')
    same_reader = (chosen == TEXT_SOURCE_VLM and _is_vlm_source(stored_source)) or (
        chosen == TEXT_SOURCE_OCR and stored_source == TEXT_SOURCE_OCR
    )
    if same_reader:
        # The same reader's reading still wins: keep its stored confidence,
        # engine and source spelling.
        target['text_source'] = stored_source
        target['text_confidence'] = view['text_confidence']
        target['text_engine_version'] = view['text_engine_version']
    return target


def changed_fields(box: RegionBox, target: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``target`` that differs from ``box`` (absent == null)."""
    return {k: v for k, v in target.items() if getattr(box, k) != v}


@dataclass
class RegionTextRepairPlan:
    """What a repair would change, for the dry-run report. ``changes`` maps
    ``crop_id`` to ``{box_id: changed attributes}``."""

    scanned: int = 0
    human_skipped: int = 0
    changes: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    text_changed: Counter[str] = field(default_factory=Counter)
    choice: Counter[str] = field(default_factory=Counter)
    vlm_invalid: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)


def _transition(box: RegionBox, target: dict[str, Any]) -> str:
    before = 'vlm' if _is_vlm_source(box.text_source) else box.text_source
    after = 'vlm' if _is_vlm_source(target.get('text_source')) else target.get('text_source')
    return f'{before or "none"} -> {after or "none"}'


def candidate_query() -> dict[str, Any]:
    """Items holding at least one box with a stored reading."""
    F = get_region_fields()
    return {
        'bool': {
            'filter': [
                box_query(
                    {
                        'bool': {
                            'should': [
                                {'exists': {'field': f'{F.boxes}.text'}},
                                {'exists': {'field': f'{F.boxes}.text_vlm'}},
                                {'exists': {'field': f'{F.boxes}.text_ocr'}},
                            ],
                            'minimum_should_match': 1,
                        }
                    },
                    F,
                )
            ]
        }
    }


async def plan_region_text_repair(
    client: Any,
    *,
    index: str,
    profile: DetectionProfile,
    rules: RegionTextRules,
    page_size: int = 500,
    max_examples: int = 5,
) -> RegionTextRepairPlan:
    """Scan ``index`` (read-only) and plan every box whose chosen text
    attributes would change. A text-free profile plans nothing."""
    F = get_region_fields()
    plan = RegionTextRepairPlan()
    if not profile.reads_text:
        return plan
    cursor: list[Any] | None = None
    while True:
        body: dict[str, Any] = {
            'size': page_size,
            'query': candidate_query(),
            'sort': [{'crop_id': 'asc'}],
            '_source': {'includes': ['crop_id', 'vlm_model', F.boxes]},
        }
        if cursor is not None:
            body['search_after'] = cursor
        resp = await client.search(index=index, body=body)
        hits = (resp.get('hits') or {}).get('hits') or []
        if not hits:
            break
        cursor = hits[-1].get('sort')
        for h in hits:
            plan.scanned += 1
            per_box: dict[str, dict[str, Any]] = {}
            src = h.get('_source') or {}
            for box in read_boxes(src, F):
                target = rederive(box, profile=profile, rules=rules, vlm_model=src.get('vlm_model'))
                if target is None:
                    if box.text_source == HUMAN_TEXT_SOURCE:
                        plan.human_skipped += 1
                    continue
                plan.choice[str(target.get('text_choice'))] += 1
                if target.get('text_vlm_invalid'):
                    plan.vlm_invalid[str(target['text_vlm_invalid'])] += 1
                diff = changed_fields(box, target)
                if not diff:
                    continue
                per_box[box.box_id] = diff
                if 'text' in diff:
                    key = _transition(box, target)
                    plan.text_changed[key] += 1
                    shown = plan.examples.setdefault(key, [])
                    if len(shown) < max_examples:
                        shown.append(
                            f'{h["_id"]}/{box.box_id}: {box.text!r} -> {target.get("text")!r} '
                            f'(vlm={stored_vlm_reading(_view(box))!r}, ocr={box.text_ocr!r})'
                        )
            if per_box:
                plan.changes[h['_id']] = per_box
        if len(hits) < page_size or cursor is None:
            break
    return plan


async def apply_region_text_repair(
    client: Any,
    plan: RegionTextRepairPlan,
    *,
    index: str,
    profile: DetectionProfile,
    rules: RegionTextRules,
) -> dict[str, Any]:
    """Rewrite the planned items under OCC, re-deriving each box from its
    current state (a box that changed since planning is re-judged; human
    text is left alone). The box list is written through
    :func:`~src.services.curation.region_boxes.boxes_write_fields`."""
    F = get_region_fields()
    now = datetime.now(UTC).isoformat()

    def _merge(_doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        boxes: list[RegionBox] = []
        changed = False
        for box in read_boxes(current, F):
            target = rederive(box, profile=profile, rules=rules, vlm_model=current.get('vlm_model'))
            diff = changed_fields(box, target) if target is not None else {}
            if diff:
                changed = True
            boxes.append(dataclasses.replace(box, **diff) if diff else box)
        if not changed:
            return {}
        return {**boxes_write_fields(boxes, current_src=current, F=F), 'updated_at': now}

    if not plan.changes:
        return {'updated': 0, 'skipped_due_to_conflict': 0, 'errors': []}
    return await occ_skip_on_conflict_bulk(
        client,
        doc_ids=list(plan.changes),
        merger=_merge,
        index=index,
        refresh='wait_for',
        writer_id=REPAIR_WRITER,
    )


__all__ = [
    'MANAGED_ATTRS',
    'REPAIR_WRITER',
    'RegionTextRepairPlan',
    'apply_region_text_repair',
    'candidate_query',
    'changed_fields',
    'plan_region_text_repair',
    'rederive',
    'stored_vlm_reading',
]
