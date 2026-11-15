"""The kinds of human decision, who may make one, and how claims are ranked.

One list: the rows of the ``claim_kinds`` table are this catalog (15a section 3.2). A claim is
one appended row; the current state of an item is the fold of the machine results and the live
claims (:mod:`src.domain.fold`). Phase-5 gallery kinds (favourite, album, person, ...) are added
by the lane that folds them; this catalog holds the kinds the item, box and duplicate folds read.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ActorKind(StrEnum):
    """Who made a claim. A person outranks an import or a combine whatever the time (R3, B2)."""

    HUMAN = 'human'
    IMPORT = 'import'
    COMBINE = 'combine'


ACTOR_RANK: dict[ActorKind, int] = {
    ActorKind.HUMAN: 2,
    ActorKind.IMPORT: 1,
    ActorKind.COMBINE: 1,
}


class SubjectKind(StrEnum):
    """What a claim is about (the ``claims.subject_kind`` check)."""

    PHOTO = 'photo'
    MEDIA = 'media'
    ITEM = 'item'
    BOX = 'box'
    PAIR = 'pair'
    FACE = 'face'
    ENTITY = 'entity'
    CLAIM = 'claim'


class ClaimKind(StrEnum):
    ITEM_DRAWN = 'item_drawn'
    GEOMETRY_SET = 'geometry_set'
    CLASS_SET = 'class_set'
    CLASS_CLEARED = 'class_cleared'
    CLASS_MOVED = 'class_moved'
    HOLDOUT_FROZEN = 'holdout_frozen'
    EXCLUDED = 'excluded'
    RESTORED = 'restored'
    REVIEW_DISMISSED = 'review_dismissed'
    REVIEW_RESTORED = 'review_restored'
    VLM_DISMISSED = 'vlm_dismissed'
    PROBE_DISMISSED = 'probe_dismissed'
    NEEDS_NEW_CLASS = 'needs_new_class'
    AUDIT_SAMPLED = 'audit_sampled'
    COMBINE_CONFLICT = 'combine_conflict'
    REGION_STATUS_SET = 'region_status_set'
    BOX_CREATED = 'box_created'
    BOX_MOVED = 'box_moved'
    BOX_STATE_SET = 'box_state_set'
    BOX_TEXT_SET = 'box_text_set'
    BOX_DELETED = 'box_deleted'
    DUPLICATE_VERDICT = 'duplicate_verdict'
    WITHDRAW = 'withdraw'


class Layer(StrEnum):
    """The fold layer a kind feeds (15a section 4.3)."""

    GEOMETRY = 'geometry'
    CLASS = 'class'
    FLAG = 'flag'
    BOX = 'box'
    REGION = 'region'
    PAIR = 'pair'
    WITHDRAW = 'withdraw'


@dataclass(frozen=True)
class KindInfo:
    kind: ClaimKind
    subject_kind: SubjectKind
    layer: Layer
    pins_geometry: bool
    """The claim records the item's box the person saw. The item's box is pinned to it only while
    the item is locked by today's ``is_locked_item`` (a person's or validated import's class,
    holdout, human or import geometry, a human delete, any human-owned box, a region set a person
    or an import validated); a clear, a move, a dismissal and a flag never pin (Q11)."""
    asserts_object: bool
    """An unmatched claim creates an item or box from the claim alone; false: it stays dormant."""


def _k(
    kind: ClaimKind,
    subject: SubjectKind,
    layer: Layer,
    *,
    pins: bool = False,
    asserts: bool = False,
) -> tuple[ClaimKind, KindInfo]:
    return kind, KindInfo(kind, subject, layer, pins, asserts)


_I, _B = SubjectKind.ITEM, SubjectKind.BOX
KINDS: dict[ClaimKind, KindInfo] = dict(
    (
        _k(ClaimKind.ITEM_DRAWN, _I, Layer.GEOMETRY, pins=True, asserts=True),
        _k(ClaimKind.GEOMETRY_SET, _I, Layer.GEOMETRY, pins=True, asserts=True),
        _k(ClaimKind.CLASS_SET, _I, Layer.CLASS, pins=True, asserts=True),
        _k(ClaimKind.CLASS_CLEARED, _I, Layer.CLASS),
        _k(ClaimKind.CLASS_MOVED, _I, Layer.CLASS),
        _k(ClaimKind.HOLDOUT_FROZEN, _I, Layer.CLASS, pins=True, asserts=True),
        _k(ClaimKind.EXCLUDED, _I, Layer.FLAG, pins=True),
        _k(ClaimKind.RESTORED, _I, Layer.FLAG),
        _k(ClaimKind.REVIEW_DISMISSED, _I, Layer.FLAG),
        _k(ClaimKind.REVIEW_RESTORED, _I, Layer.FLAG),
        _k(ClaimKind.VLM_DISMISSED, _I, Layer.FLAG),
        _k(ClaimKind.PROBE_DISMISSED, _I, Layer.FLAG),
        _k(ClaimKind.NEEDS_NEW_CLASS, _I, Layer.FLAG),
        _k(ClaimKind.AUDIT_SAMPLED, _I, Layer.FLAG),
        _k(ClaimKind.COMBINE_CONFLICT, _I, Layer.FLAG),
        _k(ClaimKind.REGION_STATUS_SET, _I, Layer.REGION, pins=True),
        _k(ClaimKind.BOX_CREATED, _B, Layer.BOX, pins=True, asserts=True),
        _k(ClaimKind.BOX_MOVED, _B, Layer.BOX, pins=True, asserts=True),
        _k(ClaimKind.BOX_STATE_SET, _B, Layer.BOX, pins=True, asserts=True),
        _k(ClaimKind.BOX_TEXT_SET, _B, Layer.BOX, pins=True, asserts=True),
        _k(ClaimKind.BOX_DELETED, _B, Layer.BOX, pins=True),
        _k(ClaimKind.DUPLICATE_VERDICT, SubjectKind.PAIR, Layer.PAIR),
        _k(ClaimKind.WITHDRAW, SubjectKind.CLAIM, Layer.WITHDRAW),
    )
)

GEOMETRY_KINDS = frozenset({ClaimKind.ITEM_DRAWN, ClaimKind.GEOMETRY_SET})
CLASS_DECIDING_KINDS = frozenset(
    {ClaimKind.CLASS_SET, ClaimKind.CLASS_CLEARED, ClaimKind.CLASS_MOVED}
)
PINNING_KINDS = frozenset(k for k, info in KINDS.items() if info.pins_geometry)


# Whether a claim of the kind can change the rank of any item among its photo's items (rank is the
# order of box areas, ``fold.rank_items``). One entry per kind, so a kind added to the catalogue and
# not classified here fails ``tests/domain/test_claim_kind_rank_effect.py``. The writer of the record
# (``src.services.record.apply``) locks and folds only the named items when no input can change rank, and
# every sibling of the photo when one can. A drawn item and a geometry claim change an area, an exclusion
# or restore is treated as changing rank (the wire counts an excluded item out of its photo's order), and
# a withdraw can undo any of those, so the kind alone cannot say: it is rank changing.
RANK_EFFECT: dict[ClaimKind, bool] = {
    ClaimKind.ITEM_DRAWN: True,
    ClaimKind.GEOMETRY_SET: True,
    ClaimKind.CLASS_SET: False,
    ClaimKind.CLASS_CLEARED: False,
    ClaimKind.CLASS_MOVED: False,
    ClaimKind.HOLDOUT_FROZEN: False,
    ClaimKind.EXCLUDED: True,
    ClaimKind.RESTORED: True,
    ClaimKind.REVIEW_DISMISSED: False,
    ClaimKind.REVIEW_RESTORED: False,
    ClaimKind.VLM_DISMISSED: False,
    ClaimKind.PROBE_DISMISSED: False,
    ClaimKind.NEEDS_NEW_CLASS: False,
    ClaimKind.AUDIT_SAMPLED: False,
    ClaimKind.COMBINE_CONFLICT: False,
    ClaimKind.REGION_STATUS_SET: False,
    ClaimKind.BOX_CREATED: False,
    ClaimKind.BOX_MOVED: False,
    ClaimKind.BOX_STATE_SET: False,
    ClaimKind.BOX_TEXT_SET: False,
    ClaimKind.BOX_DELETED: False,
    ClaimKind.DUPLICATE_VERDICT: False,
    ClaimKind.WITHDRAW: True,
}
RANK_CHANGING_KINDS = frozenset(k for k, changes in RANK_EFFECT.items() if changes)


def actor_rank(actor_kind: ActorKind) -> int:
    return ACTOR_RANK[actor_kind]
