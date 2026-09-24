"""Canonical ``RegionFields.rejection_reason`` values the pipeline writes.

The detection worker stamps one of these on every region it rejects
(``verify_rejected`` / ``detection_failed``); a human reviewer may also
write a free-text reason, which is not listed here. ``GET
{prefix}/regions/vocabulary`` serves :func:`rejection_reason_catalog` so a
client renders labels from here instead of hardcoding UI copy.

``kind`` says who decided:

* ``model_verdict`` -- the verifier looked at the box and judged it wrong.
* ``automatic`` -- a deterministic geometry check rejected the box before
  any model saw it.
* ``needs_human`` -- nothing decided: the verifier kept giving no verdict,
  so the item was parked for a person to judge.

``match`` is ``exact`` (the stored value equals ``id``) or ``prefix`` (the
stored value starts with ``id``; the remainder is the check's own reason,
e.g. ``sanity_reject:degenerate_zero_size``). A ``prefix`` entry carries a
``label_template`` whose ``{detail}`` is that remainder; ``exact`` entries
have ``label_template: null``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# The verifier said the candidate box is wrong (the region is elsewhere).
REJECT_REASON_VERIFIER = 'region_visible_elsewhere'
# The verifier gave no verdict on the candidate box on every allowed
# attempt; the box is kept as a candidate for a human to judge.
REJECT_REASON_NO_VERDICT = 'verifier_no_verdict'
# The geometry gate rejected the box; its own reason follows the prefix.
REJECT_REASON_SANITY_PREFIX = 'sanity_reject:'

REJECTION_REASON_KINDS: tuple[str, ...] = ('model_verdict', 'automatic', 'needs_human')


@dataclass(frozen=True)
class RejectionReasonInfo:
    """How one rejection reason reads to a reviewer."""

    label: str
    kind: str
    match: str = 'exact'
    label_template: str | None = None


REJECTION_REASON_INFO: dict[str, RejectionReasonInfo] = {
    REJECT_REASON_VERIFIER: RejectionReasonInfo(
        'Verifier: the box is wrong (region is elsewhere)', 'model_verdict'
    ),
    REJECT_REASON_SANITY_PREFIX: RejectionReasonInfo(
        'Box failed the geometry check',
        'automatic',
        match='prefix',
        label_template='Box failed the geometry check ({detail})',
    ),
    REJECT_REASON_NO_VERDICT: RejectionReasonInfo(
        'Verifier gave no verdict — needs human review', 'needs_human'
    ),
}


def rejection_reason_catalog() -> list[dict[str, Any]]:
    """``[{id, label, kind, match, label_template}]`` for every
    pipeline-written reason."""
    return [
        {
            'id': reason,
            'label': info.label,
            'kind': info.kind,
            'match': info.match,
            'label_template': info.label_template,
        }
        for reason, info in REJECTION_REASON_INFO.items()
    ]


__all__ = [
    'REJECTION_REASON_INFO',
    'REJECTION_REASON_KINDS',
    'REJECT_REASON_NO_VERDICT',
    'REJECT_REASON_SANITY_PREFIX',
    'REJECT_REASON_VERIFIER',
    'RejectionReasonInfo',
    'rejection_reason_catalog',
]
