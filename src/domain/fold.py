"""The fold: the current state of an item from machine results and live claims (15a section 4).

``fold_item`` is pure: no I/O, no clock, no environment. Time and ids are inputs, so the same
inputs give the same row in any order the results arrived in. Every claim layer is read before
every result layer; a machine write therefore cannot change what a person decided (R5), and the
lock rule of ``src/clients/occ_locks.py`` is the shape of the data instead of a check.

Layers: 1 geometry (R3 pin), 2 class, 3 flags, 4 boxes and region status, 5 placement, 6
revisions. A withdrawn claim leaves no trace except through the holdout rule (M1) and the
revision counters, which only grow.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from src.domain.audit_math import audit_outcome
from src.domain.claim_kinds import KINDS, PINNING_KINDS, ActorKind, ClaimKind
from src.domain.claim_layers import (
    claim_geometry,
    deciding,
    of_kinds,
    pinned_item_bbox,
    split_claims,
)
from src.domain.claim_payloads import ClassMoved, ClassSet, Excluded, ProbeDismissed, VlmDismissed
from src.domain.fold_boxes import fold_boxes
from src.domain.fold_class import ClassDecision, chosen_detection, decide_class
from src.domain.fold_locks import (
    box_is_locked,
    has_import_geometry,
    is_editor_locked,
    is_locked_class,
    is_locked_geometry,
    is_locked_item,
)
from src.domain.fold_state import Conflict, ItemFold, ItemState
from src.domain.item_geometry import NOT_AN_OBJECT, box_area_norm


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import datetime

    from src.domain.claim_layers import Split
    from src.domain.claim_payloads import Box4
    from src.domain.claims import Claim
    from src.domain.fold_state import FoldSettings
    from src.domain.results import ItemInputs, RegistrySnapshot


def _newest(claims: Iterable[Claim], kind: ClaimKind) -> Claim | None:
    return deciding(of_kinds(claims, {kind}))


def _machine_geometry(inp: ItemInputs) -> Box4 | None:
    det = chosen_detection(inp.detections)
    return None if det is None else det.bbox


def _geometry(live: Sequence[Claim]) -> tuple[Box4 | None, ActorKind | None]:
    """``(box, actor)`` of a deciding drawn or moved claim: layer 1, claims before machines."""
    box, claim = claim_geometry(live)
    return box, None if claim is None else claim.actor_kind


def _flag_exclusion(live: Sequence[Claim]) -> Claim | None:
    top = deciding(of_kinds(live, {ClaimKind.EXCLUDED, ClaimKind.RESTORED}))
    return top if top is not None and top.kind is ClaimKind.EXCLUDED else None


def _needs_new_class(inp: ItemInputs, live: Sequence[Claim]) -> bool:
    flag = _newest(live, ClaimKind.NEEDS_NEW_CLASS)
    opinion = next(
        (o for o in inp.opinions if o.source == 'vlm' and o.outcome == 'new_class_pending'), None
    )
    since: datetime | None = None
    if flag is not None:
        since = flag.at
    if opinion is not None and (since is None or opinion.created_at > since):
        since = opinion.created_at
    if since is None:
        return False
    # Q1: a person who labels the object has decided its class, so a later class_set of a person
    # clears the flag (today only resolving the new-class proposal does).
    return not any(
        c.kind is ClaimKind.CLASS_SET and c.actor_kind is ActorKind.HUMAN and c.at > since
        for c in live
    )


def _vlm_dismissed(live: Sequence[Claim], decision: ClassDecision) -> bool:
    claim = _newest(live, ClaimKind.VLM_DISMISSED)
    suggested_id, suggested_name = decision.suggestion
    if claim is None or (suggested_id is None and suggested_name is None) or decision.validated:
        return False
    payload = claim.payload
    assert isinstance(payload, VlmDismissed)
    if suggested_id is not None:
        return payload.class_id_hint == suggested_id
    return payload.class_name == suggested_name


def _audit(inp: ItemInputs, live: Sequence[Claim]) -> tuple[bool, str | None]:
    sample = _newest(live, ClaimKind.AUDIT_SAMPLED)
    if sample is None:
        return False, None
    det = chosen_detection(inp.detections)
    later = [
        c
        for c in of_kinds(live, {ClaimKind.CLASS_SET})
        if c.claim_id > sample.claim_id and c.actor_kind is ActorKind.HUMAN
    ]
    last = max(later, key=lambda c: c.claim_id, default=None)
    label = getattr(sample.payload, 'label_name', None)
    if last is None or det is None or not det.class_name or not label:
        return True, None
    payload = last.payload
    assert isinstance(payload, ClassSet)
    return True, audit_outcome(detector=det.class_name, label=label, human=payload.class_name)


def _import_ids(inp: ItemInputs, live: Sequence[Claim]) -> tuple[str, ...]:
    ids: list[str] = [c.import_id for c in live if c.import_id]
    ids += [o.import_id for o in inp.opinions if o.import_id]
    ids += [d.import_id for d in inp.detections if d.import_id]
    return tuple(dict.fromkeys(ids))


def _combine_conflict(live: Sequence[Claim]) -> bool:
    flag = _newest(live, ClaimKind.COMBINE_CONFLICT)
    return flag is not None and not any(
        c.kind is ClaimKind.CLASS_SET and c.claim_id > flag.claim_id for c in live
    )


def _revisions(inp: ItemInputs, split: Split) -> tuple[int, int]:
    """Counters that only grow: claims of the layer, their withdraws, and the machine sequence."""
    all_body = [*split.live, *split.withdrawn]
    geometry_claims = {c.claim_id for c in all_body if c.kind in PINNING_KINDS}
    box_claims = {
        c.claim_id
        for c in all_body
        if KINDS[c.kind].layer.value in ('box', 'region') or c.kind is ClaimKind.REGION_STATUS_SET
    }
    withdraws = split.withdraw_of
    geometry = len(geometry_claims) + sum(1 for t in geometry_claims if t in withdraws)
    region = len(box_claims) + sum(1 for t in box_claims if t in withdraws)
    return geometry + inp.item.geometry_seq, region + inp.item.region_seq


def _updated_at(inp: ItemInputs) -> datetime:
    times = [inp.item.created_at, *(c.at for c in inp.claims)]
    times += [d.created_at for d in inp.detections]
    times += [o.created_at for o in inp.opinions]
    if inp.region is not None and inp.region.detected_at is not None:
        times.append(inp.region.detected_at)
    return max(times)


def _conflicts(inp: ItemInputs, split: Split, top_class: Claim | None) -> tuple[Conflict, ...]:
    """Import or combine class claims that did not decide because a person's claim did (B2), or
    because the item was frozen into the holdout before they were made."""
    freeze = _newest(split.live, ClaimKind.HOLDOUT_FROZEN)
    person_decides = top_class is not None and top_class.actor_kind is ActorKind.HUMAN
    out: list[Conflict] = []
    for claim in of_kinds(split.live, {ClaimKind.CLASS_SET}):
        if claim.actor_kind is ActorKind.HUMAN or claim is top_class:
            continue
        after_freeze = freeze is not None and claim.claim_id > freeze.claim_id
        if person_decides or after_freeze:
            if claim.actor_kind is ActorKind.COMBINE:
                out.append(Conflict(inp.item.crop_id, 'combine_conflict', claim.claim_id))
            else:
                out.append(Conflict(inp.item.crop_id, 'locked_conflict', claim.claim_id))
    return tuple(out)


def fold_item(
    inp: ItemInputs, registry: RegistrySnapshot, settings: FoldSettings
) -> ItemFold | None:
    """The ``item_state`` row and ``box_state`` rows of one item, or ``None`` when the item has
    neither a current detection nor a live asserting claim (its identity row is then deleted)."""
    split = split_claims(inp.claims)
    live = split.live
    asserting = any(KINDS[c.kind].asserts_object for c in live)
    if chosen_detection(inp.detections) is None and not asserting:
        return None
    claimed_bbox, geometry_actor = _geometry(live)

    exclusion = _flag_exclusion(live)
    excluded = exclusion is not None
    excluded_reason = None if exclusion is None else _excluded_reason(exclusion)
    excluded_by_human = exclusion is not None and exclusion.actor_kind is ActorKind.HUMAN
    deleted_by_human = excluded_by_human and excluded_reason == NOT_AN_OBJECT

    decision = decide_class(inp, split, registry, settings)
    validated = decision.validated and not excluded
    holdout = any(c.kind is ClaimKind.HOLDOUT_FROZEN for c in live)
    boxes = fold_boxes(inp, live, settings)
    import_ids = _import_ids(inp, live)

    import_geom = has_import_geometry(
        import_ids=import_ids, class_source=decision.class_source, class_validated=validated
    )
    human_geom = geometry_actor is ActorKind.HUMAN or deleted_by_human
    if human_geom:
        geometry_source = 'human'
    elif geometry_actor is not None or import_geom:
        geometry_source = 'import'
    else:
        geometry_source = 'detector'

    class_locked = is_locked_class(
        class_source=decision.class_source, class_validated=validated, holdout=holdout
    )
    geometry_locked = is_locked_geometry(
        human_geometry=human_geom, import_geometry=import_geom, holdout=holdout
    )
    any_box_locked = any(
        box_is_locked(
            source=b.source,
            rejection_reason=b.rejection_reason,
            text_source=b.text_source,
            state=b.state,
        )
        for b in boxes.boxes
    )
    label_locked = is_locked_item(
        class_locked=class_locked,
        geometry_locked=geometry_locked,
        any_box_locked=any_box_locked,
        region_validated=boxes.closed,
        region_verifier=boxes.verifier,
    )

    # Q11: the item's box is pinned to the box the person saw exactly while today's is_locked_item
    # holds (a locked class, geometry, box or validated set); a clear, move, dismissal or flag
    # never pins.
    pinned = label_locked
    if claimed_bbox is not None:
        bbox: Box4 | None = claimed_bbox
    elif pinned:
        bbox = pinned_item_bbox(live) or _machine_geometry(inp)
    else:
        bbox = _machine_geometry(inp) or pinned_item_bbox(live)
    if bbox is None:
        return None

    geometry_revision, region_revision = _revisions(inp, split)
    audit_sample, audit_result = _audit(inp, live)
    review = deciding(of_kinds(live, {ClaimKind.REVIEW_DISMISSED, ClaimKind.REVIEW_RESTORED}))
    moved = deciding(of_kinds(live, {ClaimKind.CLASS_MOVED}))
    probe = _newest(live, ClaimKind.PROBE_DISMISSED)
    accepted = [b for b in boxes.boxes if b.state == 'accepted']
    rejected = [b for b in boxes.boxes if b.state == 'rejected']
    scores = [b.score for b in boxes.boxes if b.score is not None]

    ref = decision.ref
    state = ItemState(
        crop_id=inp.item.crop_id,
        photo_key=inp.item.photo_key,
        bbox=bbox,
        area=box_area_norm(bbox),
        rank=1,
        geometry_source=geometry_source,  # type: ignore[arg-type]
        geometry_revision=geometry_revision,
        class_id=None if ref is None else ref.class_id,
        class_name=None if ref is None else ref.name,
        class_source=decision.class_source,
        label_source=decision.label_source,
        class_validated=validated,
        class_unresolved=decision.unresolved,
        class_labeled_at=decision.labeled_at,
        class_confidence=decision.confidence,
        class_claim_id=None if decision.claim is None else decision.claim.claim_id,
        class_op_id=decision.op_id,
        class_opinion_source=decision.opinion_source,
        class_labeler=decision.labeler,
        class_detector=decision.detector,
        class_detector_version=decision.detector_version,
        vlm_suggestion_class_id=decision.suggestion[0],
        vlm_suggestion_name=decision.suggestion[1],
        vlm_suggestion_dismissed=_vlm_dismissed(live, decision),
        needs_new_class=_needs_new_class(inp, live),
        probe_dismissed_class=None if probe is None else _probe_class(probe),
        excluded=excluded,
        excluded_reason=excluded_reason,
        excluded_at=None if exclusion is None else exclusion.at,
        excluded_by_human=excluded_by_human,
        holdout=holdout,
        review_dismissed_at=review.at
        if review is not None and review.kind is ClaimKind.REVIEW_DISMISSED
        else None,
        audit_sample=audit_sample,
        audit_outcome=audit_result,
        combine_conflict=_combine_conflict(live),
        import_ids=import_ids,
        region_status=boxes.status,
        region_validated=boxes.closed,
        region_verifier=boxes.verifier,
        region_label_source=None,
        region_count=len(accepted),
        region_rejected_count=len(rejected),
        region_max_score=max(scores) if scores else None,
        region_set_complete=boxes.set_complete,
        region_revision=region_revision,
        region_seen_seq=boxes.generation,
        moved_partition_id=None if moved is None else _moved(moved).partition_id,
        moved_cluster_uid=None if moved is None else _moved(moved).cluster_uid,
        created_at=inp.item.created_at,
        updated_at=_updated_at(inp),
        label_locked=label_locked,
        geometry_pinned=pinned,
        geometry_locked=is_editor_locked(
            import_geometry=import_geom, holdout=holdout, deleted=deleted_by_human
        ),
    )
    top_class = deciding(
        of_kinds(live, {ClaimKind.CLASS_SET, ClaimKind.CLASS_CLEARED, ClaimKind.CLASS_MOVED})
    )
    return ItemFold(state=state, boxes=boxes.boxes, conflicts=_conflicts(inp, split, top_class))


def _excluded_reason(claim: Claim) -> str:
    payload = claim.payload
    assert isinstance(payload, Excluded)
    return payload.reason


def _probe_class(claim: Claim) -> str:
    payload = claim.payload
    assert isinstance(payload, ProbeDismissed)
    return payload.class_name


def _moved(claim: Claim) -> ClassMoved:
    payload = claim.payload
    assert isinstance(payload, ClassMoved)
    return payload


def rank_items(folds: Sequence[ItemFold]) -> list[ItemFold]:
    """Rank a photo's folded items by box area, largest first, ties by id (ingest's rule)."""
    ordered = sorted(folds, key=lambda f: (-f.state.area, f.state.crop_id))
    ranks = {f.state.crop_id: n for n, f in enumerate(ordered, start=1)}
    return [replace(f, state=replace(f.state, rank=ranks[f.state.crop_id])) for f in folds]


def moved_placement(state: ItemState, serving_partition_id: str | None) -> tuple[str, str] | None:
    """The cluster a person moved the item into, only while that partition is the serving one
    (M21): an old move never shows a cluster the serving partition does not have, and its class
    effect stays. ``None`` leaves the placement to the serving partition's membership."""
    if (
        state.moved_partition_id is None
        or state.moved_cluster_uid is None
        or state.moved_partition_id != serving_partition_id
    ):
        return None
    return state.moved_partition_id, state.moved_cluster_uid
