"""Cluster purity thresholds — shared by the auto-promote gate and the
cluster cards, so a card never calls a cluster "pure" that the gate
wouldn't promote. Leaf module.

``purity`` is the top class's share of a cluster's labelled members.
"""

from __future__ import annotations

from typing import Literal


PROMOTE_MIN_PURITY = 0.85
"""Auto-promote gate: minimum purity. Also the "pure" tier floor."""

PROMOTE_MIN_MEMBERS = 4
"""Auto-promote gate: minimum cluster size."""

PROMOTE_MIN_LABELLED_SHARE = 0.5
"""Auto-promote gate: at least this share of members must carry a class."""

PURITY_MIXED_MIN = 0.6
"""Below this a cluster is "noisy"; between it and the gate, "mixed"."""

PurityTier = Literal['pure', 'mixed', 'noisy']


def purity_tier(purity: float | None) -> PurityTier | None:
    if purity is None:
        return None
    if purity >= PROMOTE_MIN_PURITY:
        return 'pure'
    if purity >= PURITY_MIXED_MIN:
        return 'mixed'
    return 'noisy'


def is_promotable(
    *,
    members: int,
    labelled: int,
    purity: float | None,
    min_purity: float = PROMOTE_MIN_PURITY,
    min_members: int = PROMOTE_MIN_MEMBERS,
) -> bool:
    """The auto-promote gate for one cluster."""
    if purity is None or members <= 0:
        return False
    return (
        members >= min_members
        and purity >= min_purity
        and labelled / members >= PROMOTE_MIN_LABELLED_SHARE
    )


def purity_thresholds() -> dict[str, float | int]:
    """Served on ``GET /clusters`` so a client renders tiers without its
    own cut points."""
    return {
        'pure_min': PROMOTE_MIN_PURITY,
        'mixed_min': PURITY_MIXED_MIN,
        'promote_min_members': PROMOTE_MIN_MEMBERS,
        'promote_min_labelled_share': PROMOTE_MIN_LABELLED_SHARE,
    }


__all__ = [
    'PROMOTE_MIN_LABELLED_SHARE',
    'PROMOTE_MIN_MEMBERS',
    'PROMOTE_MIN_PURITY',
    'PURITY_MIXED_MIN',
    'PurityTier',
    'is_promotable',
    'purity_thresholds',
    'purity_tier',
]
