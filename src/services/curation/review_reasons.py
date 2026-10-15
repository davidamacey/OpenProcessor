"""Per-item review reasons: why an item sits in the ``mismatches`` and ``regions`` tabs.

The tab-level reason comes from
:func:`~src.services.curation.review_queries.build_tab_query`; these two refine it
per item.
"""

from __future__ import annotations

from typing import Any

from src.config.region_rejection import compose_rejection_reason
from src.config.region_state import RegionStatus
from src.domain.region_boxes import read_boxes


def mismatch_reason(src: dict[str, Any], registry_names: frozenset[str], default: str) -> str:
    """Per-item reason on the ``mismatches`` tab.

    ``vlm_unmatched`` covers more than "the VLM named something outside
    the registry": a low-confidence answer that *is* a registry class is
    routed here unapplied, and legacy rows carry no answer at all. The
    tab's generic reason was false for both. ``registry_names`` are the
    normalized active class names.
    """
    from src.services.curation.new_class_terms import normalize_term

    raw = str(src.get('vlm_raw_class') or src.get('vlm_raw_label') or '').strip()
    if not raw:
        return 'VLM gave no class answer'
    if normalize_term(raw) in registry_names:
        confidence = src.get('vlm_confidence') or 'unknown'
        return f'VLM named registry class {raw!r} at {confidence} confidence; not applied'
    return default


def region_reason(src: dict[str, Any], fields: Any, default: str) -> str:
    """Per-item reason on the ``regions`` tab (mirrors :func:`mismatch_reason`).

    A verifier-rejected candidate needs a different reason than an
    accepted-but-unreviewed box: the reviewer is confirming/reversing a
    rejection, not just validating a fresh detection.

    This used to always say "...— needs human review" even for a
    reason whose own kind is a firm model/automatic verdict, and embedded
    the raw ``region_rejection_reason`` id verbatim instead of its served
    vocabulary label -- so a box the verifier actually rejected read
    "rejected (bad detection)" next to a reason that also claimed "needs
    human review", contradicting itself. :func:`compose_rejection_reason`
    words the reason from the served rejection-reason vocabulary and only
    says "needs human review" for a ``needs_human``-kind reason.

    The rejection reason lives on the rejected box itself now
    (``region_boxes[].rejection_reason``), not an item-level scalar. The
    first ``rejected`` box's reason is used (a ``verify_rejected`` item has
    only rejected boxes, but may have several; the reviewer sees each box's
    own reason on the box).
    """
    if src.get(fields.status) != RegionStatus.VERIFY_REJECTED.value:
        return default
    why = next(
        (
            b.rejection_reason
            for b in read_boxes(src, fields)
            if b.state == 'rejected' and b.rejection_reason
        ),
        None,
    )
    if why:
        return compose_rejection_reason(why)
    return 'needs human review: verifier rejected this candidate with no reason recorded'
