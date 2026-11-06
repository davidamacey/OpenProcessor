"""The loop's event payloads: snapshots under the 7,000 byte cap, thresholds, warning codes."""

from __future__ import annotations

import pytest

from src.domain import assist_loop_events as ev
from src.domain.assist_loop_control import control_flags
from src.domain.assist_loop_metrics import loop_metrics


def _node(i: int) -> dict[str, object]:
    return {
        'path': f'r{i}.0.1.2',
        'level': 3,
        'decision': 'pass',
        'class_name': 'a-rather-long-class-name-for-padding',
        'answered': 40,
        'agree': 40,
        'bound': 0.9123,
        'eligible': 300,
        'items_propagated': 260,
        'extra': 'dropped',
    }


def test_a_nodes_event_carries_at_most_forty_nodes_and_only_the_wire_fields() -> None:
    short = [{**_node(i), 'class_name': 'car'} for i in range(100)]
    payload = ev.nodes_payload('loop', short)

    assert len(payload['nodes']) == ev.NODES_PER_EVENT
    assert 'extra' not in payload['nodes'][0]
    assert ev.size(payload) <= ev.MAX_PAYLOAD_BYTES


def test_a_list_that_would_break_the_cap_is_cut_from_the_tail() -> None:
    big = [{**_node(i), 'class_name': 'x' * 400} for i in range(40)]

    payload = ev.nodes_payload('loop', big)

    assert 0 < len(payload['nodes']) < 40
    assert ev.size(payload) <= ev.MAX_PAYLOAD_BYTES
    assert payload['nodes'][0]['path'] == 'r0.0.1.2'


def test_a_payload_that_cannot_fit_even_empty_is_an_error() -> None:
    with pytest.raises(ev.PayloadTooLargeError):
        ev.fit({'loop_id': 'x' * 8000, 'nodes': []}, 'nodes')


def test_the_iteration_event_drops_the_deepest_levels_first() -> None:
    levels = [{'level': i, 'pad': 'y' * 300} for i in range(40)]

    payload = ev.iteration_payload('loop', 1, 'closed', {'levels': levels, 'totals': {'a': 1}})

    assert ev.size(payload) <= ev.MAX_PAYLOAD_BYTES
    kept = payload['counts']['levels']
    assert kept == levels[: len(kept)]
    assert payload['counts']['totals'] == {'a': 1}


@pytest.mark.parametrize(
    ('before', 'after', 'expected'),
    [
        (None, 10, None),
        (0, 50, 50),
        (49, 51, 50),
        (50, 79, None),
        (50, 96, 95),
        (95, 100, 100),
        (100, 100, None),
    ],
)
def test_a_threshold_is_published_once_when_crossed(before, after, expected) -> None:
    assert ev.crossed(before, after) == expected


def test_percent_counts_spent_and_reserved_and_ignores_an_uncapped_unit() -> None:
    assert ev.percent(40, 10, 100) == 50
    assert ev.percent(500, 0, 100) == 100
    assert ev.percent(5, 5, None) is None


def test_only_the_declared_warning_codes_exist() -> None:
    assert ev.warning_payload('l', 'usage_missing', 'd')['code'] == 'usage_missing'
    with pytest.raises(ValueError, match='warning code'):
        ev.warning_payload('l', 'made_up', 'd')


@pytest.mark.parametrize(
    ('status', 'pausing', 'blocker', 'expected'),
    [
        (
            'queued',
            False,
            None,
            {
                'can_pause': True,
                'can_resume': False,
                'can_cancel': True,
                'can_edit_budget': True,
                'can_rollback': False,
            },
        ),
        (
            'running',
            False,
            None,
            {
                'can_pause': True,
                'can_resume': False,
                'can_cancel': True,
                'can_edit_budget': True,
                'can_rollback': False,
            },
        ),
        (
            'running',
            True,
            None,
            {
                'can_pause': False,
                'can_resume': False,
                'can_cancel': True,
                'can_edit_budget': True,
                'can_rollback': False,
            },
        ),
        (
            'paused',
            False,
            None,
            {
                'can_pause': False,
                'can_resume': True,
                'can_cancel': True,
                'can_edit_budget': True,
                'can_rollback': True,
            },
        ),
        (
            'paused',
            False,
            'project_paused',
            {
                'can_pause': False,
                'can_resume': False,
                'can_cancel': True,
                'can_edit_budget': True,
                'can_rollback': True,
            },
        ),
        (
            'cancelling',
            False,
            None,
            {
                'can_pause': False,
                'can_resume': False,
                'can_cancel': False,
                'can_edit_budget': False,
                'can_rollback': False,
            },
        ),
        (
            'completed',
            False,
            None,
            {
                'can_pause': False,
                'can_resume': False,
                'can_cancel': False,
                'can_edit_budget': False,
                'can_rollback': True,
            },
        ),
        (
            'cancelled',
            False,
            None,
            {
                'can_pause': False,
                'can_resume': False,
                'can_cancel': False,
                'can_edit_budget': False,
                'can_rollback': True,
            },
        ),
    ],
)
def test_the_control_flags_follow_the_state(status, pausing, blocker, expected) -> None:
    flags = control_flags(status, pause_requested=pausing, blocker=blocker)

    got = {k: getattr(flags, k) for k in expected}
    assert got == expected
    assert flags.state == ('pausing' if pausing else status)
    assert flags.resume_blocked_reason == (blocker if status == 'paused' else None)


def test_metrics_are_none_not_zero_when_there_is_nothing_to_divide() -> None:
    empty = loop_metrics(
        items_propagated=0, items_direct=0, crops_spent=0, wall_s=0, audit_sampled=0, audit_agreed=0
    )
    assert (empty.labels_per_hour, empty.vlm_crops_per_labeled_item, empty.audit_agreement) == (
        None,
        None,
        None,
    )
    got = loop_metrics(
        items_propagated=900,
        items_direct=100,
        crops_spent=50,
        wall_s=1800,
        audit_sampled=40,
        audit_agreed=38,
    )
    assert got.items_labeled == 1000
    assert got.labels_per_hour == 2000.0
    assert got.vlm_crops_per_labeled_item == 0.05
    assert got.audit_agreement == 0.95
