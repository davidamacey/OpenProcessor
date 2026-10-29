"""Quality flags a cluster card carries, and the thresholds that raise them.

Pure rules over numbers the card already holds, so the server decides what is
worth a reviewer's attention and the client only renders the served codes.
"""

from __future__ import annotations

from typing import Literal, TypedDict


ValidationState = Literal['none', 'partial', 'complete']

MIXED_DETECTOR_SHARE_MAX = 0.8
"""The detector's top class holds less than this share of the members that have
a detector class: the detector disagrees with itself inside the cluster."""

MIXED_DETECTOR_MIN_MEMBERS = 10
"""...and only for a cluster at least this big (a handful of members is noise)."""

LABEL_AGREEMENT_MIN = 0.9
"""Labelled members agree on a class below this share: a label disagreement."""

LABEL_DISAGREEMENT_MIN_LABELLED = 5
"""...judged only with at least this many labelled members."""


class FlagDef(TypedDict):
    id: str
    label: str
    description: str


FLAG_VOCABULARY: tuple[FlagDef, ...] = (
    {
        'id': 'mixed_detector_classes',
        'label': 'Mixed detector classes',
        'description': (
            f'The detector gave the members different classes: its top class covers '
            f'under {MIXED_DETECTOR_SHARE_MAX:.0%} of them '
            f'(clusters of at least {MIXED_DETECTOR_MIN_MEMBERS} members).'
        ),
    },
    {
        'id': 'low_geometry_purity',
        'label': 'Low geometry purity',
        'description': 'Many members sit closer to another cluster centroid than to their own.',
    },
    {
        'id': 'label_disagreement',
        'label': 'Label disagreement',
        'description': (
            f'The labelled members name different classes: the top class covers under '
            f'{LABEL_AGREEMENT_MIN:.0%} (at least {LABEL_DISAGREEMENT_MIN_LABELLED} labelled).'
        ),
    },
    {
        'id': 'promotable',
        'label': 'Promotable',
        'description': 'A candidate cluster that clears the auto-promote gate.',
    },
    {
        'id': 'needs_review',
        'label': 'Needs review',
        'description': 'At least one member is in the unified review queue.',
    },
    {
        'id': 'unvalidated',
        'label': 'Unvalidated',
        'description': 'No member has a human-validated class yet.',
    },
)


def validation_state(size: int, validated: int) -> ValidationState:
    """How much of the cluster a human has validated."""
    if size <= 0 or validated <= 0:
        return 'none'
    return 'complete' if validated >= size else 'partial'


def card_flags(
    *,
    size: int,
    detector_top_share: float | None,
    purity_tier: str | None,
    label_agreement: float | None,
    labelled_count: int,
    promotable: bool,
    needs_review_count: int,
    validation_state: ValidationState,
) -> list[str]:
    """The flag codes one card raises, in :data:`FLAG_VOCABULARY` order."""
    raised = {
        'mixed_detector_classes': (
            detector_top_share is not None
            and size >= MIXED_DETECTOR_MIN_MEMBERS
            and detector_top_share < MIXED_DETECTOR_SHARE_MAX
        ),
        'low_geometry_purity': purity_tier == 'noisy',
        'label_disagreement': (
            label_agreement is not None
            and labelled_count >= LABEL_DISAGREEMENT_MIN_LABELLED
            and label_agreement < LABEL_AGREEMENT_MIN
        ),
        'promotable': promotable,
        'needs_review': needs_review_count > 0,
        'unvalidated': validation_state == 'none',
    }
    return [f['id'] for f in FLAG_VOCABULARY if raised[f['id']]]
