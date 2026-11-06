"""Rates of an assist loop, computed from its counts (pure; the review-dependent numbers of the
research plan arrive with the review routes).

``items_labeled`` is what the loop wrote, propagated plus the sampled items the VLM answered.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoopMetrics:
    items_labeled: int
    labels_per_hour: float | None
    vlm_crops_per_labeled_item: float | None
    audit_agreement: float | None


def loop_metrics(
    *,
    items_propagated: int,
    items_direct: int,
    crops_spent: int,
    wall_s: float,
    audit_sampled: int,
    audit_agreed: int,
) -> LoopMetrics:
    labeled = items_propagated + items_direct
    return LoopMetrics(
        items_labeled=labeled,
        labels_per_hour=round(labeled * 3600 / wall_s, 2) if wall_s > 0 else None,
        vlm_crops_per_labeled_item=round(crops_spent / labeled, 4) if labeled else None,
        audit_agreement=round(audit_agreed / audit_sampled, 4) if audit_sampled else None,
    )


__all__ = ['LoopMetrics', 'loop_metrics']
