"""Re-derive stored region text under the region-text validity rules.

Rows written before :mod:`src.services.detection.region_text_rules`
existed can carry a VLM reading that is not text as their chosen
``region_text`` -- the prompt's example value ("ABC123"), a "can't read
it" word, a stock run ("999") -- even where the OCR reader stored a valid
reading. This re-runs the chooser
(:func:`~src.services.detection.region_text.resolve_region_text`) on each
row's *stored* readings -- ``region_text_vlm`` (or, for rows written
before per-reader fields, a VLM-sourced ``region_text``) and
``region_text_ocr`` -- with the deployment's rules, and rewrites the
chosen-text fields (:data:`MANAGED_ATTRS`). The per-reader readings and
``region_text_raw`` are never changed, so the repair is re-runnable.

Human-typed text (``region_text_source='human'``) is never touched.
An OCR reading's own confidence isn't stored separately, so an OCR
reading newly chosen here gets ``region_text_confidence=null``; a choice
that doesn't change keeps its stored confidence and engine.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_skip_on_conflict_bulk
from src.config import get_region_fields
from src.services.detection.region_text import (
    TEXT_SOURCE_OCR,
    TEXT_SOURCE_VLM,
    DominantTextConfig,
    DominantTextReading,
    ocr_engine_id,
    resolve_region_text,
)
from src.services.labeling.vlm_client import DEFAULT_MODEL as VLM_MODEL_ID


if TYPE_CHECKING:
    from src.config import DetectionProfile
    from src.services.detection.region_text_rules import RegionTextRules


REPAIR_WRITER = 'operator:rederive_region_text'
HUMAN_TEXT_SOURCE = 'human'

# RegionFields attributes of the chosen reading, rewritten by the repair.
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


def stored_vlm_reading(doc: dict[str, Any]) -> str | None:
    """The VLM's stored reading: ``text_vlm``, else a VLM-sourced ``text``."""
    F = get_region_fields()
    vlm = doc.get(F.text_vlm)
    if not vlm and _is_vlm_source(doc.get(F.text_source)):
        vlm = doc.get(F.text)
    return str(vlm) if vlm else None


def rederive(
    doc: dict[str, Any], *, profile: DetectionProfile, rules: RegionTextRules
) -> dict[str, Any] | None:
    """Target values (storage keys) of :data:`MANAGED_ATTRS` for ``doc``, or
    ``None`` when the row holds human text or no stored reading at all, or
    the profile does not read text (a text-free profile has no chooser to
    re-run, and stored text is left as it is)."""
    if not profile.reads_text:
        return None
    F = get_region_fields()
    stored_source = doc.get(F.text_source)
    if stored_source == HUMAN_TEXT_SOURCE:
        return None
    vlm = stored_vlm_reading(doc)
    ocr_text = doc.get(F.text_ocr)
    if not vlm and not ocr_text:
        return None
    ocr = (
        DominantTextReading(str(ocr_text), str(doc.get(F.text_raw) or ''), None, None, (), 'ok')
        if ocr_text
        else None
    )
    out = resolve_region_text(
        profile.text_reader,
        vlm_text=vlm,
        vlm_confidence=None,
        vlm_engine=VLM_MODEL_ID,
        ocr=ocr,
        ocr_engine=ocr_engine_id(profile),
        normalizer=DominantTextConfig.from_profile(profile).normalizer,
        rules=rules,
    )
    target = {getattr(F, a): out.get(a) for a in MANAGED_ATTRS}
    chosen = out.get('text_source')
    same_reader = (chosen == TEXT_SOURCE_VLM and _is_vlm_source(stored_source)) or (
        chosen == TEXT_SOURCE_OCR and stored_source == TEXT_SOURCE_OCR
    )
    if same_reader:
        # The same reader's reading still wins: keep its stored confidence,
        # engine and source spelling.
        target[F.text_source] = stored_source
        target[F.text_confidence] = doc.get(F.text_confidence)
        target[F.text_engine_version] = doc.get(F.text_engine_version)
    return target


def changed_fields(doc: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``target`` that differs from ``doc`` (absent == null)."""
    return {k: v for k, v in target.items() if doc.get(k) != v}


@dataclass
class RegionTextRepairPlan:
    """What a repair would change, for the dry-run report."""

    scanned: int = 0
    human_skipped: int = 0
    changes: dict[str, dict[str, Any]] = field(default_factory=dict)
    text_changed: Counter[str] = field(default_factory=Counter)
    choice: Counter[str] = field(default_factory=Counter)
    vlm_invalid: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)


def _transition(doc: dict[str, Any], target: dict[str, Any]) -> str:
    F = get_region_fields()
    before = 'vlm' if _is_vlm_source(doc.get(F.text_source)) else doc.get(F.text_source)
    after = 'vlm' if _is_vlm_source(target.get(F.text_source)) else target.get(F.text_source)
    return f'{before or "none"} -> {after or "none"}'


def candidate_query() -> dict[str, Any]:
    F = get_region_fields()
    return {
        'bool': {
            'should': [
                {'exists': {'field': F.text}},
                {'exists': {'field': F.text_vlm}},
                {'exists': {'field': F.text_ocr}},
            ],
            'minimum_should_match': 1,
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
    """Scan ``index`` (read-only) and plan every row whose chosen text
    fields would change. A text-free profile plans nothing."""
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
            '_source': {
                'includes': [
                    'crop_id',
                    *(getattr(F, a) for a in (*MANAGED_ATTRS, 'text_vlm', 'text_ocr', 'text_raw')),
                ]
            },
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
            doc = h.get('_source') or {}
            target = rederive(doc, profile=profile, rules=rules)
            if target is None:
                if doc.get(F.text_source) == HUMAN_TEXT_SOURCE:
                    plan.human_skipped += 1
                continue
            plan.choice[str(target.get(F.text_choice))] += 1
            if target.get(F.text_vlm_invalid):
                plan.vlm_invalid[str(target[F.text_vlm_invalid])] += 1
            diff = changed_fields(doc, target)
            if not diff:
                continue
            plan.changes[h['_id']] = diff
            if F.text in diff:
                key = _transition(doc, target)
                plan.text_changed[key] += 1
                shown = plan.examples.setdefault(key, [])
                if len(shown) < max_examples:
                    shown.append(
                        f'{h["_id"]}: {doc.get(F.text)!r} -> {target.get(F.text)!r} '
                        f'(vlm={stored_vlm_reading(doc)!r}, ocr={doc.get(F.text_ocr)!r})'
                    )
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
    """Rewrite the planned rows under OCC, re-deriving from each row's
    current state (a row that changed since planning is re-judged; human
    text is left alone)."""
    now = datetime.now(UTC).isoformat()

    def _merge(_doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        target = rederive(current, profile=profile, rules=rules)
        diff = changed_fields(current, target) if target is not None else {}
        return {**diff, 'updated_at': now} if diff else {}

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
