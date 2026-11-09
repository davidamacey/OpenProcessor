"""Cluster purity thresholds of the auto-promote gate. Leaf module.

The auto-promote gate's *label purity* is the top class's share of a cluster's
labelled members (served on cards as ``label_agreement``).
"""

from __future__ import annotations


PROMOTE_MIN_PURITY = 0.85
"""Auto-promote gate: minimum purity. Also."""

PROMOTE_MIN_MEMBERS = 4
"""Auto-promote gate: minimum cluster size."""

PROMOTE_MIN_PRECISION = 0.95
"""Auto-promote audit gate: audited detector precision a class needs
(``POST /clusters/auto_promote``, ``promote_min_precision``)."""

PROMOTE_MIN_LABELLED_SHARE = 0.5
"""Auto-promote gate: at least this share of members must carry a class."""


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


__all__ = [
    'PROMOTE_MIN_LABELLED_SHARE',
    'PROMOTE_MIN_MEMBERS',
    'PROMOTE_MIN_PURITY',
    'is_promotable',
]
