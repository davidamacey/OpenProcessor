"""The accuracy audit's pure parts: stratified allocation, outcome, Wilson interval
and the report, each against a hand-computed table."""

from __future__ import annotations

import pytest

from src.services.curation.audit_math import (
    AuditRow,
    allocate_sample,
    audit_outcome,
    build_report,
    wilson_interval,
)


def test_allocation_fills_every_stratum_when_the_budget_is_large() -> None:
    sizes = {'a': 100, 'b': 10, 'c': 3}
    assert allocate_sample(sizes, min_per_class=5, total=300) == sizes


def test_allocation_under_a_small_budget_levels_the_floors_round_robin() -> None:
    # Rounds of one each: a1 b1 c1 | a2 b2 c2 | a3 b3 c3 (c is full) | the last one goes to a.
    alloc = allocate_sample({'a': 100, 'b': 10, 'c': 3}, min_per_class=5, total=10)
    assert alloc == {'a': 4, 'b': 3, 'c': 3}
    assert sum(alloc.values()) == 10


def test_allocation_spreads_the_surplus_over_the_spare_capacity() -> None:
    # Floors 10 + 10; 40 left over spare 90 and 40: quotas 27.69 and 12.31 -> 28 and 12.
    alloc = allocate_sample({'a': 100, 'b': 50}, min_per_class=10, total=60)
    assert alloc == {'a': 38, 'b': 22}


def test_allocation_never_exceeds_a_stratum_or_the_budget() -> None:
    sizes = {'a': 2, 'b': 40, 'c': 7}
    for total in range(0, 60, 7):
        alloc = allocate_sample(sizes, min_per_class=5, total=total)
        assert all(0 <= alloc[k] <= sizes[k] for k in sizes)
        assert sum(alloc.values()) == min(total, sum(sizes.values()))


def test_allocation_is_deterministic() -> None:
    sizes = {'x': 9, 'y': 9, 'z': 9}
    assert allocate_sample(sizes, min_per_class=2, total=7) == allocate_sample(
        dict(reversed(sizes.items())), min_per_class=2, total=7
    )


@pytest.mark.parametrize(
    ('detector', 'label', 'human', 'expected'),
    [
        ('car', 'car', 'car', 'agree'),
        ('car', 'truck', 'car', 'vlm_wrong'),
        ('truck', 'car', 'car', 'detector_wrong'),
        ('bus', 'truck', 'car', 'both_wrong'),
        ('Traffic Light', 'traffic_light', 'traffic_light', 'agree'),  # names compare normalized
    ],
)
def test_outcome_table(detector: str, label: str, human: str, expected: str) -> None:
    assert audit_outcome(detector=detector, label=label, human=human) == expected


def test_wilson_interval_matches_the_reference_values() -> None:
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3)
    assert high == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)  # no evidence: the whole range
    low, high = wilson_interval(10, 10)
    assert high == pytest.approx(1.0)
    assert low == pytest.approx(0.7225, abs=1e-3)


def _row(detector: str, label: str, human: str, *, vlm: bool = True) -> AuditRow:
    return AuditRow(
        detector=detector,
        label=label,
        label_is_vlm=vlm,
        human=human,
        outcome=audit_outcome(detector=detector, label=label, human=human),
    )


ROWS = [
    _row('car', 'car', 'car'),
    _row('car', 'car', 'car'),
    _row('car', 'truck', 'car'),  # vlm_wrong
    _row('car', 'truck', 'truck'),  # detector_wrong
    _row('bus', 'truck', 'car', vlm=False),  # both_wrong, label from a classifier
]


def test_report_per_class_precision_intervals_and_confusion() -> None:
    report = build_report(ROWS, min_per_class=3, pending=2)
    detector = {c['name']: c for c in report['detector']}
    assert (detector['car']['n'], detector['car']['correct']) == (4, 3)
    assert detector['car']['precision'] == pytest.approx(0.75)
    assert (detector['car']['ci_low'], detector['car']['ci_high']) == pytest.approx(
        wilson_interval(3, 4)
    )
    assert detector['car']['insufficient_sample'] is False
    assert (detector['bus']['n'], detector['bus']['correct']) == (1, 0)
    assert detector['bus']['insufficient_sample'] is True  # 1 < 3

    # VLM precision is over the VLM-labelled rows only, grouped by the VLM's class.
    vlm = {c['name']: c for c in report['vlm']}
    assert (vlm['car']['n'], vlm['car']['correct']) == (2, 2)
    assert (vlm['truck']['n'], vlm['truck']['correct']) == (2, 1)
    assert 'bus' not in vlm

    assert report['confusion'] == {'car': {'car': 3, 'truck': 1}, 'bus': {'car': 1}}
    assert report['outcomes'] == {
        'agree': 2,
        'vlm_wrong': 1,
        'detector_wrong': 1,
        'both_wrong': 1,
    }
    assert (report['audited'], report['pending']) == (5, 2)


def test_an_empty_report_is_empty_not_an_error() -> None:
    report = build_report([], min_per_class=30, pending=0)
    assert report['detector'] == []
    assert report['vlm'] == []
    assert report['confusion'] == {}
    assert report['audited'] == 0
