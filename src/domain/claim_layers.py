"""Which claims are live, and which one decides a layer (15a section 4.3, "Ranking").

Shared by the subject matching and the fold, so the two never read a different geometry for the
same claim group. A claim is live until a ``withdraw`` names it; a withdraw is never withdrawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.domain.claim_kinds import GEOMETRY_KINDS, PINNING_KINDS, ClaimKind


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.domain.claim_payloads import Box4
    from src.domain.claims import Claim


@dataclass(frozen=True)
class Split:
    live: tuple[Claim, ...]
    """Claims no withdraw names, in claim order (withdraws themselves are not listed)."""
    withdrawn: tuple[Claim, ...]
    """Claims a withdraw names, in claim order."""
    withdraw_of: dict[int, Claim]
    """Target claim id to the withdraw row naming it."""


def split_claims(claims: Iterable[Claim]) -> Split:
    ordered = sorted(claims, key=lambda c: c.claim_id)
    withdraw_of = {
        c.withdraws_claim_id: c
        for c in ordered
        if c.kind is ClaimKind.WITHDRAW and c.withdraws_claim_id is not None
    }
    body = [c for c in ordered if c.kind is not ClaimKind.WITHDRAW]
    return Split(
        live=tuple(c for c in body if c.claim_id not in withdraw_of),
        withdrawn=tuple(c for c in body if c.claim_id in withdraw_of),
        withdraw_of=withdraw_of,
    )


def deciding(claims: Iterable[Claim]) -> Claim | None:
    """The claim that wins its layer: a person before an import or combine, then the newest."""
    return max(claims, key=lambda c: c.order, default=None)


def of_kinds(claims: Iterable[Claim], kinds: frozenset[ClaimKind] | set[ClaimKind]) -> list[Claim]:
    return [c for c in claims if c.kind in kinds]


def seen_item_bbox(claim: Claim) -> Box4 | None:
    """The item's box the person saw when they made the claim: the subject box of an item claim,
    the recorded ``item_bbox`` of a box or region claim."""
    seen = getattr(claim.payload, 'item_bbox', None)
    return seen if seen is not None else claim.subject.bbox


def claim_geometry(live: Sequence[Claim]) -> tuple[Box4 | None, Claim | None]:
    """The box a deciding drawn or moved claim gives an item, and that claim (layer 1);
    ``(None, None)`` leaves the box to the pin or to the machine."""
    geometry = deciding(of_kinds(live, GEOMETRY_KINDS))
    if geometry is not None:
        box = getattr(geometry.payload, 'bbox', None)
        return (box if box is not None else geometry.subject.bbox), geometry
    return None, None


def pinned_item_bbox(live: Sequence[Claim]) -> Box4 | None:
    """The item's box as shown at the newest claim that pins (used while pinned): whoever made it,
    the box in effect when the item was last pinned is the one that stays."""
    pins = of_kinds(live, PINNING_KINDS)
    return seen_item_bbox(max(pins, key=lambda c: c.claim_id)) if pins else None


def newest_machine_bbox(claims: Sequence[Claim]) -> Box4 | None:
    """The detector box of the newest claim that recorded one."""
    for claim in sorted(claims, key=lambda c: c.claim_id, reverse=True):
        if claim.subject.machine_bbox is not None:
            return claim.subject.machine_bbox
    return None
