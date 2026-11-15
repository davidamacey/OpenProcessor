"""Finding a claim's subject again after every machine result was rebuilt (15a section 3.4).

Pure and shared by the fold, re-detect and import. Claims are matched per item, not per claim
(M14): all live claims of one item (its ``subject_crop_id``) move together.

1. Group a photo's live claims by the item they name. A group made in another decode frame than
   the photo's now matches nothing and is reported ``frame_mismatch`` (fail closed).
2. A detection with the group's own id takes the group.
3. Else groups and detections are paired greedily, best overlap first, at ``LABEL_IOU_MATCH`` or
   better (ties by id): the detection against the group's box (layer 1) or the detector box the
   person saw. A matched item keeps the group's id, so a touched item keeps its wire id across
   detector versions.
4. A group with no match and an asserting claim becomes an item from its claims; a group of
   dormant kinds only creates nothing and waits for a later detection, so a deleted object stays
   deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.domain.box_overlap import LABEL_IOU_MATCH, iou
from src.domain.claim_kinds import KINDS, SubjectKind
from src.domain.claim_layers import claim_geometry, newest_machine_bbox, split_claims


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from src.domain.claim_payloads import Box4
    from src.domain.claims import Claim


@dataclass(frozen=True)
class DetectionRef:
    crop_id: str
    bbox: Box4


@dataclass(frozen=True)
class BoxRef:
    box_id: str
    bbox: Box4


@dataclass(frozen=True)
class MatchResult:
    matched: dict[str, str]
    """Group id (the item keeps it) to the detection that took the group."""
    created: tuple[str, ...]
    """Groups with no detection and a live asserting claim: items made from the claims."""
    dormant: tuple[str, ...]
    """Groups with no detection and no asserting claim: nothing is created, the claims wait."""
    frame_mismatch: tuple[str, ...]
    new_detections: tuple[str, ...]
    """Detections no group took: plain machine items."""


def _greedy(
    pairs: Iterable[tuple[float, str, str]], *, taken_left: set[str], taken_right: set[str]
) -> dict[str, str]:
    """Best overlap first; ties by (left id, right id): the same answer in any input order."""
    out: dict[str, str] = {}
    for _neg, left, right in sorted(pairs):
        if left in taken_left or right in taken_right:
            continue
        taken_left.add(left)
        taken_right.add(right)
        out[left] = right
    return out


def _best_overlap(candidates: Iterable[Box4 | None], box: Box4) -> float:
    return max((iou(c, box) for c in candidates if c is not None), default=0.0)


def match_items(
    photo_frame: int, detections: Sequence[DetectionRef], claims: Iterable[Claim]
) -> MatchResult:
    """Match one photo's item and box claims (withdraws included) to its detections."""
    split = split_claims(claims)
    groups: dict[str, list[Claim]] = {}
    for claim in split.live:
        if claim.subject.kind in (SubjectKind.ITEM, SubjectKind.BOX) and claim.subject.crop_id:
            groups.setdefault(claim.subject.crop_id, []).append(claim)

    mismatch = sorted(
        gid for gid, group in groups.items() if group[-1].subject.frame != photo_frame
    )
    usable = {gid: g for gid, g in groups.items() if gid not in mismatch}

    taken_groups: set[str] = set()
    taken_dets: set[str] = set()
    matched: dict[str, str] = {}
    for det in detections:
        if det.crop_id in usable and det.crop_id not in taken_groups:
            matched[det.crop_id] = det.crop_id
            taken_groups.add(det.crop_id)
            taken_dets.add(det.crop_id)

    pairs: list[tuple[float, str, str]] = []
    for gid, group in usable.items():
        if gid in taken_groups:
            continue
        effective, _ = claim_geometry(group)
        # Question 5: 15a lists the effective box and the machine box only, which leaves a group of
        # dormant kinds (a lone dismissal, a delete) no box to match by; the box the newest claim
        # saw is added so a deleted object stays deleted when its detection comes back moved.
        candidates = (effective, newest_machine_bbox(group), group[-1].subject.bbox)
        for det in detections:
            if det.crop_id in taken_dets:
                continue
            value = _best_overlap(candidates, det.bbox)
            if value >= LABEL_IOU_MATCH:
                pairs.append((-value, gid, det.crop_id))
    matched.update(_greedy(pairs, taken_left=taken_groups, taken_right=taken_dets))

    created: list[str] = []
    dormant: list[str] = []
    for gid, group in sorted(usable.items()):
        if gid in matched:
            continue
        asserts = any(KINDS[c.kind].asserts_object for c in group)
        (created if asserts else dormant).append(gid)
    return MatchResult(
        matched=matched,
        created=tuple(created),
        dormant=tuple(dormant),
        frame_mismatch=tuple(mismatch),
        new_detections=tuple(sorted(d.crop_id for d in detections if d.crop_id not in taken_dets)),
    )


def match_boxes(
    claim_boxes: Mapping[str, Sequence[Box4 | None]], machine: Sequence[BoxRef]
) -> dict[str, str]:
    """Claim box id to the machine box it applies to: the same id first, then the best overlap
    of the claimed box (as shown, or as moved) against a machine box, at ``LABEL_IOU_MATCH``.
    A split (one claimed box, two new boxes each under the threshold) matches nothing."""
    machine_ids = {m.box_id for m in machine}
    taken_claims: set[str] = set()
    taken_machine: set[str] = set()
    out = {bid: bid for bid in claim_boxes if bid in machine_ids}
    taken_claims.update(out)
    taken_machine.update(out.values())
    pairs = [
        (-value, bid, m.box_id)
        for bid, boxes in claim_boxes.items()
        if bid not in taken_claims
        for m in machine
        if m.box_id not in taken_machine
        if (value := _best_overlap(boxes, m.bbox)) >= LABEL_IOU_MATCH
    ]
    out.update(_greedy(pairs, taken_left=taken_claims, taken_right=taken_machine))
    return out
