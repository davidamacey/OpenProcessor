"""Layer 2 of the fold: the effective class of an item (15a sections 4.2 and 4.3).

A person's class claim decides before any machine opinion; with no deciding claim the machine
class is the first present opinion in the project's precedence order. Every opinion stays a
row, so a different order is a re-fold, never a data change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from src.domain.claim_kinds import CLASS_DECIDING_KINDS, ActorKind, ClaimKind
from src.domain.claim_layers import deciding, of_kinds, split_claims
from src.domain.claim_payloads import ClassSet, HoldoutFrozen, MachineClass
from src.domain.ingest_class_sources import (
    DETECTOR_CLASS_SOURCE,
    HUMAN_CLASS_SOURCE,
    LABEL_IMPORT_CLASS_SOURCE,
    OPEN_VOCAB_CLASS_SOURCE,
    OPEN_VOCAB_TARGET_CLASS_SOURCE,
    PROPAGATED_CLASS_SOURCE,
    VLM_CLASS_SOURCE,
)


if TYPE_CHECKING:
    from datetime import datetime

    from src.domain.claim_layers import Split
    from src.domain.claims import Claim
    from src.domain.fold_state import FoldSettings
    from src.domain.results import (
        ClassOpinion,
        ClassRef,
        DetectionResult,
        ItemInputs,
        RegistrySnapshot,
    )

VLM_UNMATCHED = 'vlm_unmatched'
VLM_NEW_CLASS_PENDING = 'vlm_new_class_pending'
VLM_RECLASSIFIED = 'vlm_reclassified'
UNLABELED_PROPOSAL = 'unlabeled_proposal'

_VLM_WIRE_SOURCE = {
    'matched': VLM_CLASS_SOURCE,
    'reclassified': VLM_RECLASSIFIED,
    'unmatched': VLM_UNMATCHED,
    'new_class_pending': VLM_NEW_CLASS_PENDING,
}


@dataclass(frozen=True)
class ClassDecision:
    ref: ClassRef | None
    class_source: str | None
    label_source: str | None = None
    validated: bool = False
    unresolved: bool = False
    labeled_at: datetime | None = None
    confidence: float | None = None
    claim: Claim | None = None
    op_id: str | None = None
    opinion_source: str | None = None
    labeler: str | None = None
    detector: str | None = None
    detector_version: str | None = None
    suggestion: tuple[int | None, str | None] = (None, None)
    by_freeze: bool = False


def chosen_detection(
    detections: tuple[DetectionResult, ...],
    after: datetime | None = None,
    before: datetime | None = None,
) -> DetectionResult | None:
    """The detection that gives the item its machine box: the newest detector row; an
    open-vocabulary row only without a detector row; an import's only without either."""
    live = [
        d
        for d in detections
        if (after is None or d.created_at > after) and (before is None or d.created_at <= before)
    ]
    for kind in ('detector', 'open_vocab', 'import'):
        group = [d for d in live if d.analyser_kind == kind]
        if group:
            return max(group, key=lambda d: d.created_at)
    return None


def detection_source(det: DetectionResult, settings: FoldSettings) -> str:
    if det.analyser_kind == 'detector':
        return f'{settings.primary_detector_name}_proposal'
    if det.analyser_kind == 'open_vocab':
        return OPEN_VOCAB_CLASS_SOURCE
    return UNLABELED_PROPOSAL


def opinion_source(op: ClassOpinion) -> str:
    """The wire ``class_source`` an opinion shows."""
    if op.source == 'vlm':
        return _VLM_WIRE_SOURCE.get(op.outcome, VLM_CLASS_SOURCE)
    if op.source == 'classifier':
        return f'{op.analyser_name}_{"low_conf" if op.outcome == "low_conf" else "model"}'
    return {
        'open_vocab_target': OPEN_VOCAB_TARGET_CLASS_SOURCE,
        'import_suggestion': LABEL_IMPORT_CLASS_SOURCE,
        'assist_propagated': PROPAGATED_CLASS_SOURCE,
        'adopted_detector': DETECTOR_CLASS_SOURCE,
    }[op.source]


def _vlm_suggestion(
    op: ClassOpinion, ref: ClassRef | None, source: str
) -> tuple[int | None, str | None]:
    if source in (VLM_CLASS_SOURCE, VLM_RECLASSIFIED) and ref is not None:
        return ref.class_id, ref.name
    if source == VLM_NEW_CLASS_PENDING:
        return None, op.proposed_class or None
    return None, None


def machine_class(
    inp: ItemInputs,
    settings: FoldSettings,
    registry: RegistrySnapshot,
    *,
    after: datetime | None = None,
    before: datetime | None = None,
    precedence: tuple[str, ...] | None = None,
) -> ClassDecision:
    """The first present machine opinion of the precedence order, ignoring every opinion and
    detection created at or before ``after`` (a person's clear hides only what came before it).
    An opinion row without a class never displaces one with a class; an empty VLM answer is no
    opinion (M17)."""
    order = settings.precedence if precedence is None else precedence
    opinions: dict[str, ClassOpinion] = {
        o.source: o
        for o in inp.opinions
        if (after is None or o.created_at > after) and (before is None or o.created_at <= before)
    }
    for position, source in enumerate(order):
        if source == 'detection':
            det = chosen_detection(inp.detections, after, before)
            if det is not None:
                return ClassDecision(None, detection_source(det, settings))
            continue
        op = opinions.get(source)
        if op is None or op.outcome == 'empty':
            continue
        if op.class_name is None and op.outcome not in ('unmatched', 'new_class_pending'):
            continue
        ref = registry.resolve(op.class_name, op.class_id_hint) if op.class_name else None
        wire = opinion_source(op)
        if op.outcome == 'new_class_pending':
            # Today the item keeps the class it had; only its source and suggestion change.
            below = machine_class(
                inp,
                settings,
                registry,
                after=after,
                before=before,
                precedence=order[position + 1 :],
            )
            return replace(
                below,
                class_source=wire,
                suggestion=_vlm_suggestion(op, None, wire),
                opinion_source=op.source,
                op_id=op.op_id,
            )
        suggestion = _vlm_suggestion(op, ref, wire)
        vlm = opinions.get('vlm')
        if source == 'open_vocab_target' and vlm is not None and vlm.proposed_class:
            # The target class stays; the VLM's answer is kept as a name-only suggestion.
            suggestion = (None, vlm.proposed_class)
        return ClassDecision(
            ref,
            wire,
            confidence=op.confidence,
            op_id=op.op_id,
            opinion_source=op.source,
            suggestion=suggestion,
        )
    return ClassDecision(None, None)


def _from_claim(claim: Claim, registry: RegistrySnapshot) -> ClassDecision:
    payload = claim.payload
    assert isinstance(payload, ClassSet)
    ref = registry.resolve(payload.class_name, payload.class_id_hint)
    return ClassDecision(
        ref,
        HUMAN_CLASS_SOURCE if claim.actor_kind is ActorKind.HUMAN else LABEL_IMPORT_CLASS_SOURCE,
        label_source=payload.label_source,
        validated=ref is not None,
        unresolved=ref is None,
        labeled_at=claim.at,
        claim=claim,
        op_id=claim.op_id,
        labeler=payload.labeler,
        detector=payload.profile.detector,
        detector_version=payload.profile.detector_version,
    )


def _frozen_class_claims(split: Split, freeze: Claim) -> list[Claim]:
    """Class claims written before the freeze and withdrawn after it: the freeze ignores that
    withdraw (M1), so they still count."""
    return [
        c
        for c in split.withdrawn
        if c.kind in CLASS_DECIDING_KINDS
        and c.claim_id < freeze.claim_id
        and split.withdraw_of[c.claim_id].claim_id > freeze.claim_id
    ]


def _bound_by_freeze(layer: list[Claim], freeze: Claim) -> list[Claim]:
    """Question 3: the plan binds machines and undo to the frozen class and lets a person's newer
    claim decide, but says nothing of an import. Today an import is refused on a held-out item
    (``item_labels.human_owned``) unless the class is the import's own to correct, so an import
    or combine class claim made after the freeze counts only in that case."""
    settled = [c for c in layer if c.claim_id < freeze.claim_id or c.actor_kind is ActorKind.HUMAN]
    base = deciding(settled)
    own_import = (
        base is not None
        and base.kind is ClaimKind.CLASS_SET
        and base.actor_kind is ActorKind.IMPORT
    )
    later = [c for c in layer if c not in settled]
    return settled + (later if own_import else [])


def decide_class(
    inp: ItemInputs, split: Split, registry: RegistrySnapshot, settings: FoldSettings
) -> ClassDecision:
    """Layer 2. The deciding claim among ``class_set``, ``class_cleared`` and ``class_moved``;
    otherwise the frozen class (while held out) or the machine class. ``validated`` here is
    "a resolved class_set decides"; the fold clears it while the item is excluded (M2)."""
    layer = of_kinds(split.live, CLASS_DECIDING_KINDS)
    # The class is frozen as it was at the first live freeze; a later freeze changes nothing.
    freezes = of_kinds(split.live, {ClaimKind.HOLDOUT_FROZEN})
    freeze = min(freezes, key=lambda c: c.claim_id, default=None)
    if freeze is not None:
        layer = _bound_by_freeze(layer + _frozen_class_claims(split, freeze), freeze)
    top = deciding(layer)
    if freeze is not None and (
        top is None or (top.kind is not ClaimKind.CLASS_SET and top.claim_id < freeze.claim_id)
    ):
        # A clear or a move older than the freeze left the class the freeze recorded.
        return _frozen(inp, freeze, registry, settings)
    # Machine results that arrive after a freeze never relabel the frozen item (R3).
    before = None if freeze is None else freeze.at
    if top is None or top.kind is ClaimKind.CLASS_MOVED:
        return machine_class(inp, settings, registry, before=before)
    if top.kind is ClaimKind.CLASS_CLEARED:
        return machine_class(inp, settings, registry, after=top.at, before=before)
    return _from_claim(top, registry)


def _frozen(
    inp: ItemInputs, freeze: Claim, registry: RegistrySnapshot, settings: FoldSettings
) -> ClassDecision:
    payload = freeze.payload
    assert isinstance(payload, HoldoutFrozen)
    mc = payload.machine_class
    det = chosen_detection(inp.detections)
    if mc is None:
        return ClassDecision(
            None, None if det is None else detection_source(det, settings), by_freeze=True
        )
    ref = registry.resolve(mc.class_name)
    suggestion: tuple[int | None, str | None] = (
        (ref.class_id, ref.name)
        if ref is not None and mc.source in (VLM_CLASS_SOURCE, VLM_RECLASSIFIED)
        else (None, None)
    )
    vlm = next((o for o in inp.opinions if o.source == 'vlm' and o.proposed_class), None)
    if mc.source == OPEN_VOCAB_TARGET_CLASS_SOURCE and vlm is not None:
        suggestion = (None, vlm.proposed_class)
    return ClassDecision(
        ref, mc.source, confidence=mc.confidence, suggestion=suggestion, by_freeze=True
    )


def freeze_machine_class(
    inp: ItemInputs, registry: RegistrySnapshot, settings: FoldSettings
) -> MachineClass | None:
    """The machine class the item shows now under any class claim: what a holdout freeze records
    so that the item keeps it (``None``: frozen with no class, M1). A person's clear hides what
    came before it, so the freeze records what is left."""
    split = split_claims(inp.claims)
    top = deciding(of_kinds(split.live, CLASS_DECIDING_KINDS))
    cleared_at = top.at if top is not None and top.kind is ClaimKind.CLASS_CLEARED else None
    pick = machine_class(inp, settings, registry, after=cleared_at)
    if pick.ref is None or pick.class_source is None:
        return None
    return MachineClass(
        class_name=pick.ref.name, source=pick.class_source, confidence=pick.confidence
    )
