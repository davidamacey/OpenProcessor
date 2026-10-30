"""Cluster card flags and validation state: every threshold boundary."""

from __future__ import annotations

import pytest

from src.domain import cluster_flags as cf


def _flags(**over: object) -> list[str]:
    base: dict[str, object] = {
        'size': 50,
        'detector_top_share': 1.0,
        'purity_tier': 'pure',
        'label_agreement': 1.0,
        'labelled_count': 50,
        'promotable': False,
        'needs_review_count': 0,
        'validation_state': 'complete',
    }
    base.update(over)
    return cf.card_flags(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ('size', 'validated', 'state'),
    [
        (10, 0, 'none'),
        (10, 1, 'partial'),
        (10, 9, 'partial'),
        (10, 10, 'complete'),
        (0, 0, 'none'),
    ],
)
def test_validation_state(size: int, validated: int, state: str) -> None:
    assert cf.validation_state(size, validated) == state


def test_a_clean_cluster_carries_no_flag() -> None:
    assert _flags() == []


@pytest.mark.parametrize(
    ('share', 'size', 'flagged'),
    [
        (0.79, 10, True),
        (0.8, 10, False),  # at the floor: not mixed
        (0.5, 9, False),  # under the member floor
        (None, 50, False),  # no detector class at all
    ],
)
def test_mixed_detector_classes_boundaries(share: float | None, size: int, flagged: bool) -> None:
    assert ('mixed_detector_classes' in _flags(detector_top_share=share, size=size)) is flagged


@pytest.mark.parametrize(('tier', 'flagged'), [('noisy', True), ('mixed', False), (None, False)])
def test_low_geometry_purity_is_the_noisy_tier(tier: str | None, flagged: bool) -> None:
    assert ('low_geometry_purity' in _flags(purity_tier=tier)) is flagged


@pytest.mark.parametrize(
    ('agreement', 'labelled', 'flagged'),
    [
        (0.89, 5, True),
        (0.9, 5, False),  # at the floor
        (0.5, 4, False),  # under the labelled floor
        (None, 0, False),
    ],
)
def test_label_disagreement_boundaries(
    agreement: float | None, labelled: int, flagged: bool
) -> None:
    got = _flags(label_agreement=agreement, labelled_count=labelled)
    assert ('label_disagreement' in got) is flagged


def test_promotable_needs_review_and_unvalidated_flags() -> None:
    assert _flags(promotable=True) == ['promotable']
    assert _flags(needs_review_count=1) == ['needs_review']
    assert _flags(needs_review_count=0) == []
    assert _flags(validation_state='none') == ['unvalidated']
    assert _flags(validation_state='partial') == []


def test_flags_come_out_in_vocabulary_order() -> None:
    got = _flags(
        detector_top_share=0.1,
        purity_tier='noisy',
        label_agreement=0.1,
        promotable=True,
        needs_review_count=3,
        validation_state='none',
    )
    assert got == [f['id'] for f in cf.FLAG_VOCABULARY]
    assert len(got) == 6


def test_vocabulary_entries_are_complete_and_unique() -> None:
    ids = [f['id'] for f in cf.FLAG_VOCABULARY]
    assert len(set(ids)) == len(ids)
    for entry in cf.FLAG_VOCABULARY:
        assert entry['label'].strip()
        assert entry['description'].strip()
