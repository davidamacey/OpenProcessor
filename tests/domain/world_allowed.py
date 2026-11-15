"""The differences between today's code and the fold that the plan decided on purpose.

Each entry names the difference and the plan line that decided it. ``applies`` answers, for one
field that differs after one act, whether this entry explains it; it looks at the whole history
and at the state of both worlds, never only at the two values. An entry is narrow on purpose: a
difference it does not explain is a bug in the fold or a gap in the plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config.region_state import RegionStatus
from src.domain.box_overlap import iou
from src.domain.claim_kinds import KINDS, ActorKind, ClaimKind
from src.domain.claim_layers import split_claims
from src.domain.region_boxes import RegionBox, derive_status
from tests.domain.fold_support import PRIMARY


if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.domain.world_compare import Ctx

# Plan 15 section 14, answer 5, written out again here: the oracle must not share the list
# the fold uses, or a wrong list would explain its own difference.
ORACLE_ORDER = (
    'open_vocab_target',
    'vlm',
    'import_suggestion',
    'assist_propagated',
    'classifier',
    'adopted_detector',
    'detection',
)
CLASS_FIELDS = frozenset(
    {
        'p_is_locked_class',
        'class_id',
        'class_name',
        'class_source',
        'class_validated',
        'label_validated',
        'vlm_suggestion',
    }
)
LOCK_FIELDS = frozenset(
    {
        'geometry_source',
        'geometry_locked',
        'label_locked',
        'p_has_human_geometry',
        'p_has_import_geometry',
        'p_is_locked_geometry',
        'p_is_locked_item',
        'p_is_human_deleted',
    }
)
BOX_DERIVED = frozenset(
    {
        'region_boxes',
        'region_status',
        'region_count',
        'region_rejected_count',
        'region_validated',
        'label_validated',
        'region_set_complete',
        'label_locked',
        'p_is_locked_box',
    }
)


@dataclass(frozen=True)
class Allowed:
    name: str
    cite: str
    applies: Callable[[Ctx, str], bool]


def _live(ctx: Ctx):
    return split_claims(ctx.fold.log.claims).live


def _wire_source(source: str, outcome: str) -> str:
    return {
        'vlm': {
            'matched': 'vlm',
            'unmatched': 'vlm_unmatched',
            'new_class_pending': 'vlm_new_class_pending',
        }.get(outcome, 'vlm'),
        'classifier': f'{PRIMARY}_{"low_conf" if outcome == "low_conf" else "model"}',
        'open_vocab_target': 'open_vocab_target',
        'import_suggestion': 'external_label',
        'assist_propagated': 'assist_propagated',
        'adopted_detector': 'detector',
    }[source]


def _last_clear(ctx: Ctx):
    cleared = [c for c in _live(ctx) if c.kind is ClaimKind.CLASS_CLEARED]
    return max((c.at for c in cleared), default=None)


def _expected_machine_class(
    ctx: Ctx, skip: frozenset[str] = frozenset()
) -> tuple[str | None, str | None]:
    """Independent oracle of 15a 4.2: ``(wire source, class name)`` of the first source of the
    precedence order that holds a usable opinion made after the last clear."""
    after = _last_clear(ctx)
    freeze = max((c.at for c in _live(ctx) if c.kind is ClaimKind.HOLDOUT_FROZEN), default=None)
    for source in ORACLE_ORDER:
        if source in skip:
            continue
        if source == 'detection':
            return (
                (f'{PRIMARY}_proposal', None)
                if after is None or any(d.created_at > after for d in ctx.fold.detections)
                else (None, None)
            )
        op = ctx.fold.opinions.get(source)
        if op is None or (after is not None and op.created_at <= after) or op.outcome == 'empty':
            continue
        if freeze is not None and op.created_at > freeze:
            continue
        if op.class_name is None and op.outcome not in ('unmatched', 'new_class_pending'):
            continue
        if op.outcome == 'new_class_pending':
            below = _expected_machine_class(
                ctx, skip | {source, *ORACLE_ORDER[: ORACLE_ORDER.index(source)]}
            )
            return 'vlm_new_class_pending', below[1]
        return _wire_source(source, op.outcome), op.class_name
    return None, None


def _machine_decided(ctx: Ctx) -> bool:
    s = ctx.fold.fold().state
    return s.class_claim_id is None


def _doc_machine_decided(ctx: Ctx) -> bool:
    d = ctx.doc_view
    source = d['class_source'] or ''
    return 'human' not in source and not (source == 'external_label' and d['class_validated'])


def _fold_matches_oracle(ctx: Ctx) -> bool:
    state = ctx.fold.fold().state
    freeze = next((c for c in _live(ctx) if c.kind is ClaimKind.HOLDOUT_FROZEN), None)
    if state.holdout and freeze is not None:
        # A frozen item shows what the freeze recorded; the row it came from may be replaced.
        recorded = getattr(freeze.payload, 'machine_class', None)
        if recorded is None:
            return ctx.fold_view['class_name'] is None
        return (ctx.fold_view['class_source'], ctx.fold_view['class_name']) == (
            recorded.source,
            recorded.class_name,
        )
    source, name = _expected_machine_class(ctx)
    return ctx.fold_view['class_source'] == source and ctx.fold_view['class_name'] == name


def _precedence(ctx: Ctx, field: str) -> bool:
    d, f = ctx.doc_view, ctx.fold_view
    return (
        field in CLASS_FIELDS | {'needs_new_class', 'label_source', 'refusal'}
        and _machine_decided(ctx)
        and _doc_machine_decided(ctx)
        and (d['class_source'], d['class_name']) != (f['class_source'], f['class_name'])
        and _fold_matches_oracle(ctx)
    )


def _acted_on_other_machine_class(ctx: Ctx, field: str) -> bool:
    """A bulk accept takes the machine class of the moment; under the fixed order that class is
    another one, so the accept lands in one world only, or takes another class."""
    if field not in CLASS_FIELDS | LOCK_FIELDS | {'label_source'}:
        return False
    fold_accepted = any(
        c.kind is ClaimKind.CLASS_SET and getattr(c.payload, 'via', None) == 'bulk_accept'
        for c in _live(ctx)
    )
    doc_accepted = ctx.doc.doc.get('label_source') == 'human_bulk_accept'
    return any(a[0] == 'accept' for a in ctx.acts) and (
        fold_accepted != doc_accepted
        or (fold_accepted and ctx.doc_view['class_name'] != ctx.fold_view['class_name'])
    )


def _kept_while_locked(ctx: Ctx, field: str) -> bool:
    dropped = set(ctx.doc.dropped_machine_sources)
    if field in {'needs_new_class', 'vlm_suggestion'}:
        return 'vlm' in dropped
    if field not in CLASS_FIELDS or ctx.fold.fold().state.class_claim_id is not None:
        return False
    wire = {_wire_source(s, o) for s in dropped if s in ORACLE_ORDER for o in ('matched', 'model')}
    return ctx.fold_view['class_source'] in wire


def _exclusion_keeps_import_class(ctx: Ctx, field: str) -> bool:
    s = ctx.fold.fold().state
    return (
        field in CLASS_FIELDS | LOCK_FIELDS | {'label_source', 'refusal', 'bbox_norm'}
        and s.excluded
        and ctx.fold_view['class_source'] == 'external_label'
        and (ctx.doc_view['class_source'] != 'external_label' or field == 'bbox_norm')
    )


def _needs_follows_opinion(ctx: Ctx, field: str) -> bool:
    vlm = ctx.fold.opinions.get('vlm')
    flags = [c for c in _live(ctx) if c.kind is ClaimKind.NEEDS_NEW_CLASS]
    cleared = _last_clear(ctx)
    if vlm is not None and cleared is not None and vlm.created_at <= cleared:
        return (field == 'vlm_suggestion' and ctx.fold_view['vlm_suggestion'] == (None, None)) or (
            field == 'needs_new_class' and not ctx.fold_view['needs_new_class']
        )
    if vlm is None or flags:
        return False
    if field == 'needs_new_class':
        return (
            ctx.doc_view['needs_new_class']
            and not ctx.fold_view['needs_new_class']
            and vlm.outcome != 'new_class_pending'
        )
    if field == 'vlm_suggestion':  # a stale proposed name of an earlier answer
        return (
            ctx.doc_view['vlm_suggestion'][1] is not None
            and ctx.fold_view['vlm_suggestion'] == (None, None)
            and vlm.proposed_class is None
        )
    return False


def _claim_only_box(ctx: Ctx, field: str) -> bool:
    if field != 'region_boxes':
        return False
    doc, fold = ctx.doc_view['region_boxes'], ctx.fold_view['region_boxes']
    mask = [
        (*d[:4], None if f[4] is None else d[4], *d[5:]) for d, f in zip(doc, fold, strict=False)
    ]
    return len(doc) == len(fold) and mask == [tuple(f) for f in fold]


def _frozen_no_name_suggestion(ctx: Ctx, field: str) -> bool:
    s = ctx.fold.fold().state
    return (
        field in {'vlm_suggestion', 'needs_new_class'}
        and s.holdout
        and ctx.fold_view['vlm_suggestion'] == (None, None)
    )


def _box_claim_follows_rerun(ctx: Ctx, field: str) -> bool:
    if field not in BOX_DERIVED | {'refusal'}:
        return False
    doc, fold = ctx.doc_view['region_boxes'], ctx.fold_view['region_boxes']
    if len(fold) >= len(doc):
        return False
    moved = any(c.kind is ClaimKind.BOX_MOVED for c in _live(ctx))  # applied to the box as shown
    return moved or all(any(iou(d[1], f[1]) >= 0.5 for f in fold) for d in doc)


def _owned_by_any_claim(ctx: Ctx, field: str) -> bool:
    return (
        field == 'p_is_human_owned_item'
        and ctx.fold_view[field]
        and not ctx.doc_view[field]
        and any(c.actor_kind is ActorKind.HUMAN for c in _live(ctx))
    )


def _import_wakes_after_undo(ctx: Ctx, field: str) -> bool:
    s = ctx.fold.fold().state
    return (
        field in CLASS_FIELDS | LOCK_FIELDS | {'label_source', 'bbox_norm'}
        and ctx.doc.refused_imports > 0
        and s.class_claim_id is not None
        and s.class_source == 'external_label'
        and ctx.doc_view['class_source'] != 'external_label'
    )


def _protected_box_state(ctx: Ctx, field: str) -> bool:
    if field not in BOX_DERIVED | {'refusal'}:
        return False
    claimed = [
        c.subject.bbox for c in _live(ctx) if c.kind is ClaimKind.BOX_STATE_SET and c.subject.bbox
    ]
    doc, fold = ctx.doc_view['region_boxes'], ctx.fold_view['region_boxes']
    if not claimed or len(doc) != len(fold):
        return False
    differs = [(d, f) for d, f in zip(doc, fold, strict=True) if d != f]
    return bool(differs) and all(
        d[1] == f[1] and d[2] != f[2] and any(iou(f[1], c) >= 0.5 for c in claimed)
        for d, f in differs
    )


def _frozen_without_class(ctx: Ctx, field: str) -> bool:
    s = ctx.fold.fold().state
    return (
        field in {'class_source', 'vlm_suggestion', 'needs_new_class'}
        and s.holdout
        and s.class_id is None
        and ctx.doc_view['class_id'] is None
    )


def _clear_then_new_detection(ctx: Ctx, field: str) -> bool:
    after = _last_clear(ctx)
    return (
        field == 'class_source'
        and after is not None
        and ctx.doc_view['class_source'] is None
        and ctx.fold_view['class_source'] == f'{PRIMARY}_proposal'
        and any(d.created_at > after for d in ctx.fold.detections)
    )


def _clear_pins_box(ctx: Ctx, field: str) -> bool:
    pins = [c for c in _live(ctx) if KINDS[c.kind].pins_geometry]
    return (
        field == 'bbox_norm'
        and any(c.kind is ClaimKind.CLASS_CLEARED and c.actor_kind is ActorKind.HUMAN for c in pins)
        and not ctx.doc.last_bbox_refused_by_lock
    )


def _deleted_box_stays_deleted(ctx: Ctx, field: str) -> bool:
    if field not in BOX_DERIVED:
        return False
    deleted = [c.subject.bbox for c in _live(ctx) if c.kind is ClaimKind.BOX_DELETED]
    fold_boxes = {b[1] for b in ctx.fold_view['region_boxes']}
    extra = [b[1] for b in ctx.doc_view['region_boxes'] if b[1] not in fold_boxes]
    return bool(extra) and all(any(iou(e, d) >= 0.5 for d in deleted if d) for e in extra)


def _reopened_by_proposed(ctx: Ctx, field: str) -> bool:
    d, f = ctx.doc_view, ctx.fold_view
    if field == 'refusal':  # the fold applied the act that today's refused, on the reopened set
        return d['region_validated'] and any(b[2] == 'proposed' for b in d['region_boxes'])
    return (
        field in BOX_DERIVED | {'label_validated', 'label_locked', 'refusal'}
        and d['region_validated']
        and not f['region_validated']
        and any(b[2] == 'proposed' for b in f['region_boxes'])
        and any(b[2] == 'proposed' for b in d['region_boxes'])
    )


def _human_clear_beats_import(ctx: Ctx, field: str) -> bool:
    if field not in CLASS_FIELDS | LOCK_FIELDS:
        return False
    live = _live(ctx)
    human_class = [
        c
        for c in live
        if c.kind in (ClaimKind.CLASS_CLEARED, ClaimKind.CLASS_MOVED)
        and c.actor_kind is ActorKind.HUMAN
    ]
    imports = [
        c for c in live if c.kind is ClaimKind.CLASS_SET and c.actor_kind is ActorKind.IMPORT
    ]
    return bool(human_class and imports) and ctx.doc_view['class_source'] == 'external_label'


def _holdout_vs_undo(ctx: Ctx, field: str) -> bool:
    s = ctx.fold.fold().state
    return (
        field in CLASS_FIELDS | LOCK_FIELDS | {'label_source'}
        and s.holdout
        and any(a[0] == 'unlabel' for a in ctx.acts)
    )


def _undo_only_own_claims(ctx: Ctx, field: str) -> bool:
    return field == 'review_dismissed' and any(a[0] == 'unlabel' for a in ctx.acts)


def _excluded_label(ctx: Ctx, field: str) -> bool:
    s = ctx.fold.fold().state
    return (
        field in CLASS_FIELDS | LOCK_FIELDS
        and s.excluded
        and (ctx.doc_view['class_validated'] or ctx.doc_view['class_source'] == 'external_label')
    )


def _import_attribution(ctx: Ctx, field: str) -> bool:
    return (
        field == 'import_ids'
        and 'i1' in ctx.fold_view['import_ids']
        and 'i1' not in ctx.doc_view['import_ids']
        and ctx.doc.refused_imports > 0
    )


def _whole_set_locks(ctx: Ctx, field: str) -> bool:
    d, f = ctx.doc_view, ctx.fold_view
    return (
        field in {'label_locked', 'p_is_locked_item'}
        and f[field]
        and not d[field]
        and d['region_validated']
        and f['region_validated']
    )


def _empty_set_verdict(ctx: Ctx, field: str) -> bool:
    d, f = ctx.doc_view, ctx.fold_view
    return (
        field == 'region_status'
        and f['region_status'] == 'no_region_visible'
        and d['region_status'] in ('false_positive', 'verify_rejected')
        and not d['region_boxes']
        and not f['region_boxes']
    )


def _set_complete(ctx: Ctx, field: str) -> bool:
    return (
        field == 'region_set_complete' and ctx.fold_view[field] is True and not ctx.doc_view[field]
    )


def _box_claims_do_not_pin(ctx: Ctx, field: str) -> bool:
    return field == 'bbox_norm' and ctx.doc.last_bbox_refused_by_lock


def _stale_status_after_drop(ctx: Ctx, field: str) -> bool:
    stubs = [
        RegionBox(box_id=b[0], bbox_norm=b[1], state=b[2]) for b in ctx.fold_view['region_boxes']
    ]
    derived = derive_status(stubs, empty_status=RegionStatus.NO_REGION_VISIBLE).value
    return (
        field == 'region_status'
        and any(a[0] == 'geometry' for a in ctx.acts)
        and ctx.fold_view['region_status'] == derived
    )


def _restore_after_exclusion_label(ctx: Ctx, field: str) -> bool:
    live = _live(ctx)
    excluded = [c.claim_id for c in live if c.kind is ClaimKind.EXCLUDED]
    labels = [c.claim_id for c in live if c.kind is ClaimKind.CLASS_SET]
    restored = [c.claim_id for c in live if c.kind is ClaimKind.RESTORED]
    return (
        field in CLASS_FIELDS | LOCK_FIELDS | {'bbox_norm'}
        and any(e < lab for e in excluded for lab in labels)
        and any(lab < r for lab in labels for r in restored)
    )


def _drop_closes_empty_set(ctx: Ctx, field: str) -> bool:
    d, f = ctx.doc_view, ctx.fold_view
    return (
        field in BOX_DERIVED | {'refusal'}
        and any(c.kind is ClaimKind.BOX_DELETED for c in _live(ctx))
        and f['region_validated']
        and not d['region_validated']
        and not any(b[2] == 'proposed' for b in f['region_boxes'])
    )


ALLOWED = [
    Allowed(
        'a box state a person set, with no other trace, is protected from a re-run',
        '15a 4.4 last paragraph ("a human accept of a machine box is now a claim, so no machine writer can touch it") and 4.3 layer 4',
        _protected_box_state,
    ),
    Allowed(
        'a withdrawn claim of a person lets the import label it overruled decide',
        '15a 3.5 Undo ("the current state is the fold without that claim") with 4.3 ranking (B2)',
        _import_wakes_after_undo,
    ),
    Allowed(
        'any live claim of a person keeps the item on an import undo',
        '15a 4.4 last row: is_human_owned_item is "the item has a live claim with actor_kind = human"',
        _owned_by_any_claim,
    ),
    Allowed(
        "a person's claim on a re-run box applies to the new box, not to a kept duplicate",
        '15a 3.4 step 7 (boxes matched on id, then IoU 0.5) and 4.3 layer 4',
        _box_claim_follows_rerun,
    ),
    Allowed(
        'needs_new_class follows the current VLM answer',
        '15a 4.3 layer 3 "needs_new_class = a live flag or opinion outcome new_class_pending"; opinions are one row per source',
        _needs_follows_opinion,
    ),
    Allowed(
        'a claimed box whose machine row was replaced keeps only what the claim says',
        '15a 2.4 box_state "box_key NULL: a box that exists only through a claim"; 3.4 step 7',
        _claim_only_box,
    ),
    Allowed(
        'a freeze records the class and source, not the VLM name-only suggestion',
        '15a 3.3 holdout_frozen payload machine_class {class_name, source, confidence}',
        _frozen_no_name_suggestion,
    ),
    Allowed(
        'an exclusion does not unlock an imported class for machines',
        '15a 4.3 layers 2 and 3 (the class claim decides; excluded only clears class_validated, M2) and plan 15 R5',
        _exclusion_keeps_import_class,
    ),
    Allowed(
        "a geometry edit that drops boxes leaves today's status stale",
        '15a 4.3 layer 4 "region_status ... derive_status(boxes)" (the status follows the boxes)',
        _stale_status_after_drop,
    ),
    Allowed(
        'a restore does not bring back a stale prior validation',
        '15a 5.2 excluded_prior_class_validated "deleted: the fold computes class_validated from the class claim and the exclusion" (M2)',
        _restore_after_exclusion_label,
    ),
    Allowed(
        'a bulk accept takes another machine class under the fixed order',
        '15a 4.2 (fixed precedence) with 3.2 class_set via bulk_accept; the accept rule is accept_merger on the folded class',
        _acted_on_other_machine_class,
    ),
    Allowed(
        'a freeze with no class shows no opinion',
        '15a 4.3 layer 2 holdout rule: machine_class null is "frozen as none ... no opinion applies" (M1)',
        _frozen_without_class,
    ),
    Allowed(
        'after a clear, a newer detection shows its proposal',
        '15a 4.3 layer 2 class_cleared "opinions and detections created after its at ... a later opinion shows" (M17)',
        _clear_then_new_detection,
    ),
    Allowed(
        'a clear pins the box the person saw',
        '15a 3.2 (class_cleared pins) and 4.3 layer 1; plan 15 R3',
        _clear_pins_box,
    ),
    Allowed(
        'a deleted box stays deleted when a re-run finds it again',
        '15a 3.4 step 7 (boxes matched on id then IoU) and 3.2 box_deleted "never creates"; 4.3 layer 4',
        _deleted_box_stays_deleted,
    ),
    Allowed(
        'a box set back to proposed reopens the set',
        '15a 4.3 layer 4 "closed when no box is proposed" (B3: partial review not validated)',
        _reopened_by_proposed,
    ),
    Allowed(
        'machine class by fixed precedence instead of last write',
        '15a 4.2 and plan 15 section 14 answer 5 (the order of sources); 4.3 layer 2 "Otherwise machine_class"',
        _precedence,
    ),
    Allowed(
        "a machine opinion refused by today's lock is stored, and shows once the claims go",
        'plan 15 R2, R4, R5 (a machine writer cannot conflict with a person; results are rows); 15a 4.3 layer 2',
        _kept_while_locked,
    ),
    Allowed(
        "a person's clear outranks a later import label",
        'plan 15 R3 "Claims are ranked by who made them before time: a person\'s claim beats an import\'s"; 15a 4.3 ranking (B2)',
        _human_clear_beats_import,
    ),
    Allowed(
        'a freeze holds the class against a per-item undo',
        '15a 4.3 layer 2 holdout rule (M1) and 3.5 outcome kept_holdout',
        _holdout_vs_undo,
    ),
    Allowed(
        'an undo withdraws only its own claims, a later dismissal stays',
        '15a 3.5 Undo ("a withdraw row per claim", no stored prior state); plan 15 4.2 Undo',
        _undo_only_own_claims,
    ),
    Allowed(
        'a label on an excluded item is not validated',
        '15a 4.3 layer 2 "class_validated = a resolved deciding class_set and not excluded" (M2)',
        _excluded_label,
    ),
    Allowed(
        'an import that lost to a person is still attributed',
        '15a 4.3 layer 3 import_ids "from live claims, not-rebuildable opinions" (M13); 3.5 Import',
        _import_attribution,
    ),
    Allowed(
        'a whole-set verdict closes the set, and a closed set locks the item',
        "15a 4.3 layer 4 (B3: closed set, region_verifier from the claim's actor); 4.4 is_locked_item",
        _whole_set_locks,
    ),
    Allowed(
        'a whole-set verdict over no boxes reads no_region_visible',
        '15a 3.2 region_status_set "only the empty-list verdict no_region_visible"; 3.3 expansion',
        _empty_set_verdict,
    ),
    Allowed(
        'region_set_complete follows the closed set',
        '15a 2.4 (item_state has no set_complete column; region_validated = closed set, 4.3 layer 4)',
        _set_complete,
    ),
    Allowed(
        "a person's region box or delete does not pin the item box",
        '15a 3.2 (box_* and excluded pin no geometry) and plan 15 R3 (the pin is class, validation, holdout, geometry)',
        _box_claims_do_not_pin,
    ),
    Allowed(
        'deleting every box, even by a geometry edit, closes the empty set',
        '15a 4.3 layer 4 "C the newest live human or import box or region claim" (box_deleted is one)',
        _drop_closes_empty_set,
    ),
]
