"""What a person did, as the claims to append: the pure half of every human write path.

Each function takes what the person saw (:class:`ItemView`) and what they asked for, and returns
:class:`~src.domain.claims.ClaimDraft` rows. Nothing is written here and nothing is read: the
repository inserts the drafts with a claim id and a time, under the item lock, then folds. The
guards that refuse a request (unknown class, stale revision) stay with the route.

Paths (15a section 3.2): single and batch label, bulk accept, new-class resolve, unlabel
(a clear, or the step-back undo of the newest operation), discard, move, exclude and restore,
review and suggestion dismissals, flag a new class, audit sample, holdout freeze and its
forgetting, duplicate verdict; boxes and geometry are in :mod:`src.domain.claim_builders_region`,
imports and combine in :mod:`src.domain.claim_builders_import`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.domain.claim_kinds import ActorKind, ClaimKind, SubjectKind
from src.domain.claim_payloads import (
    AuditSampled,
    ClassMoved,
    ClassProfile,
    ClassSet,
    DuplicateVerdict,
    Empty,
    Excluded,
    HoldoutFrozen,
    MachineClass,
    NeedsNewClass,
    Payload,
    ProbeDismissed,
    VlmDismissed,
)
from src.domain.claims import ClaimDraft, Subject


if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

    from src.domain.claim_payloads import Box4
    from src.domain.fold_state import ItemState

HUMAN_ACTOR = 'api'
HUMAN_LABELER = 'human'
STEP_BACK_OPERATION_KINDS = frozenset({'label', 'batch_label', 'move', 'resolve', 'discard'})
"""Operations the per-item undo steps back over: the writers of today's ``HUMAN_LABEL_WRITERS``.
A bulk accept is not among them (question 4)."""


@dataclass(frozen=True)
class ItemView:
    """The item as the person saw it: its wire id, photo, the geometry shown and the detector's."""

    crop_id: str
    photo_key: int
    frame: int
    bbox: Box4
    machine_bbox: Box4 | None = None

    def subject(self) -> Subject:
        return Subject(
            SubjectKind.ITEM,
            photo_key=self.photo_key,
            crop_id=self.crop_id,
            bbox=self.bbox,
            machine_bbox=self.machine_bbox,
            frame=self.frame,
        )


def human_draft(
    kind: ClaimKind,
    view: ItemView,
    payload: Payload,
    *,
    op_id: str | None,
    actor_kind: ActorKind = ActorKind.HUMAN,
    actor: str = HUMAN_ACTOR,
) -> ClaimDraft:
    return ClaimDraft(actor_kind, actor, op_id, kind, view.subject(), payload)


def label(
    view: ItemView,
    *,
    class_name: str,
    class_id_hint: int | None,
    label_source: str = 'human',
    via: str = 'single',
    op_id: str,
    profile: ClassProfile | None = None,
) -> list[ClaimDraft]:
    """``PUT /crops/{id}/label`` (``via='single'``) and ``PUT /crops/batch_label`` (``'batch'``),
    per item; the new-class resolve (``'resolve'``) and the bulk accept (``'bulk_accept'``)."""
    payload = ClassSet(
        class_name=class_name,
        class_id_hint=class_id_hint,
        label_source=label_source,
        via=via,
        labeler=HUMAN_LABELER,
        profile=profile or ClassProfile(),
    )
    return [human_draft(ClaimKind.CLASS_SET, view, payload, op_id=op_id)]


def resolve_new_class(
    view: ItemView, *, class_name: str, class_id_hint: int | None, op_id: str
) -> list[ClaimDraft]:
    """``POST /review/new_class_proposals/resolve``: a label that also settles the proposal."""
    return label(
        view,
        class_name=class_name,
        class_id_hint=class_id_hint,
        label_source='new_class_proposal',
        via='resolve',
        op_id=op_id,
    )


def clear_class(view: ItemView, *, op_id: str) -> list[ClaimDraft]:
    """``DELETE /crops/{id}/label`` on an item with no human decision to step back over."""
    return [human_draft(ClaimKind.CLASS_CLEARED, view, Empty(), op_id=op_id)]


@dataclass(frozen=True)
class LiveOperation:
    """One operation's live claims on one item, as the repository reads them from the log."""

    op_id: str
    kind: str
    claim_ids: tuple[int, ...]


def withdraw(claim_id: int, view: ItemView, *, op_id: str | None) -> ClaimDraft:
    del view
    return ClaimDraft(
        ActorKind.HUMAN,
        HUMAN_ACTOR,
        op_id,
        ClaimKind.WITHDRAW,
        Subject(SubjectKind.CLAIM),
        Empty(),
        withdraws_claim_id=claim_id,
    )


def withdraw_operation(
    view: ItemView, operation: LiveOperation, *, op_id: str | None
) -> list[ClaimDraft]:
    """The undo of one operation on one item: a withdraw of every live claim it made there. A
    whole operation is undone by running this over its items in chunks (B4)."""
    return [withdraw(c, view, op_id=op_id) for c in sorted(operation.claim_ids)]


def unlabel(view: ItemView, history: Sequence[LiveOperation], *, op_id: str) -> list[ClaimDraft]:
    """``DELETE /crops/{id}/label``: step back over the item's newest live label operation (a
    discard comes back whole); with none, clear the class. ``history`` is oldest first."""
    for operation in reversed(history):
        if operation.kind in STEP_BACK_OPERATION_KINDS and operation.claim_ids:
            return withdraw_operation(view, operation, op_id=op_id)
    return clear_class(view, op_id=op_id)


def discard(
    view: ItemView, *, clear_class_too: bool, dismiss_from_review: bool, op_id: str
) -> list[ClaimDraft]:
    """``POST /crops/{id}/discard``: clear the class, dismiss from review, or both, as one
    operation so the undo brings them back together."""
    drafts: list[ClaimDraft] = []
    if clear_class_too:
        drafts += clear_class(view, op_id=op_id)
    if dismiss_from_review:
        drafts.append(human_draft(ClaimKind.REVIEW_DISMISSED, view, Empty(), op_id=op_id))
    return drafts


def move_to_cluster(
    view: ItemView, *, partition_id: str, cluster_uid: str, op_id: str
) -> list[ClaimDraft]:
    """``POST /crops/move``: placement, and the class stops being a person's."""
    payload = ClassMoved(partition_id=partition_id, cluster_uid=cluster_uid)
    return [human_draft(ClaimKind.CLASS_MOVED, view, payload, op_id=op_id)]


def exclude(view: ItemView, *, reason: str, op_id: str) -> list[ClaimDraft]:
    return [human_draft(ClaimKind.EXCLUDED, view, Excluded(reason=reason), op_id=op_id)]


def restore(view: ItemView, *, op_id: str) -> list[ClaimDraft]:
    """``POST /crops/batch_unexclude``: the one way back from an exclusion (M2)."""
    return [human_draft(ClaimKind.RESTORED, view, Empty(), op_id=op_id)]


def review_dismiss(view: ItemView, *, op_id: str | None = None) -> list[ClaimDraft]:
    return [human_draft(ClaimKind.REVIEW_DISMISSED, view, Empty(), op_id=op_id)]


def review_undismiss(view: ItemView, *, op_id: str | None = None) -> list[ClaimDraft]:
    return [human_draft(ClaimKind.REVIEW_RESTORED, view, Empty(), op_id=op_id)]


def vlm_dismiss(
    view: ItemView, *, class_name: str | None, class_id_hint: int | None, op_id: str | None = None
) -> list[ClaimDraft]:
    """``POST /crops/{id}/vlm_dismiss``: the suggestion that was shown and rejected."""
    payload = VlmDismissed(class_name=class_name, class_id_hint=class_id_hint)
    return [human_draft(ClaimKind.VLM_DISMISSED, view, payload, op_id=op_id)]


def probe_dismiss(
    view: ItemView, *, class_name: str, probe_analyser_id: int, op_id: str | None = None
) -> list[ClaimDraft]:
    payload = ProbeDismissed(class_name=class_name, probe_analyser_id=probe_analyser_id)
    return [human_draft(ClaimKind.PROBE_DISMISSED, view, payload, op_id=op_id)]


def flag_new_class(
    view: ItemView, *, note: str | None, op_id: str | None = None
) -> list[ClaimDraft]:
    return [human_draft(ClaimKind.NEEDS_NEW_CLASS, view, NeedsNewClass(note=note), op_id=op_id)]


def audit_sample(
    view: ItemView, *, batch_id: str, label_name: str, label_source: str, op_id: str
) -> list[ClaimDraft]:
    payload = AuditSampled(batch_id=batch_id, label_name=label_name, label_source=label_source)
    return [human_draft(ClaimKind.AUDIT_SAMPLED, view, payload, op_id=op_id)]


def holdout_freeze(
    view: ItemView, *, freeze_id: str, machine_class: MachineClass | None, op_id: str
) -> list[ClaimDraft]:
    """``POST /test_holdout/freeze``, per frozen item. ``machine_class`` is the machine class
    that wins at the freeze, or ``None`` for an item frozen with no class (M1)."""
    payload = HoldoutFrozen(freeze_id=freeze_id, machine_class=machine_class)
    return [human_draft(ClaimKind.HOLDOUT_FROZEN, view, payload, op_id=op_id)]


def forget_holdout_freeze(
    view: ItemView, freeze_claim_ids: Iterable[int], *, op_id: str
) -> list[ClaimDraft]:
    """``forget_holdout_freeze``: the freeze is withdrawn; nothing else moves."""
    return [withdraw(c, view, op_id=op_id) for c in sorted(freeze_claim_ids)]


def duplicate_verdict(
    first: str, second: str, *, level: str, decision: str, photo_key: int | None = None
) -> list[ClaimDraft]:
    """``POST /near_duplicates/decisions``: a verdict on a pair; the ``DELETE`` is a withdraw."""
    subject = Subject(SubjectKind.PAIR, photo_key=photo_key, crop_id=first, pair_id=second)
    payload = DuplicateVerdict(level=level, decision=decision)
    return [
        ClaimDraft(
            ActorKind.HUMAN, HUMAN_ACTOR, None, ClaimKind.DUPLICATE_VERDICT, subject, payload
        )
    ]


def accept_machine_eligible(
    state: ItemState,
    *,
    min_confidence: float,
    detector_sources: Collection[str],
    vlm_sources: Collection[str],
) -> bool:
    """The rule of ``accept_merger`` on a folded item: a bulk accept takes a machine class that is
    not a person's, not validated, not held out, not excluded, not dismissed from review, not in
    the audit, and whose source reached ``min_confidence`` (``class_confidence`` is the score
    of the deciding opinion; the repository stores a VLM category as its number)."""
    if (
        state.class_id is None
        or state.class_validated
        or state.holdout
        or state.excluded
        or state.review_dismissed_at is not None
        or state.audit_sample
        or state.class_source is None
        or state.class_source.startswith('human')
        or (state.class_source == 'external_label' and state.class_validated)
    ):
        return False
    if state.class_source in detector_sources or state.class_source in vlm_sources:
        score = state.class_confidence
        return score is not None and score >= min_confidence
    return False


def accept_machine(
    view: ItemView, *, class_name: str, class_id_hint: int | None, op_id: str
) -> list[ClaimDraft]:
    """``POST /label_ops/accept_machine``, per eligible item: the machine class becomes the
    person's, with the label source that says it was accepted in bulk."""
    return label(
        view,
        class_name=class_name,
        class_id_hint=class_id_hint,
        label_source='human_bulk_accept',
        via='bulk_accept',
        op_id=op_id,
    )
