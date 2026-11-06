"""A node's ``reason`` code is served with a sentence for it, so the client renders what the
backend decided instead of keeping its own wording."""

from __future__ import annotations

import pytest

from src.db.repos.assist_loop_record import NodeRow
from src.services.curation.assist_loop.views import _node


def _row(decision: str, reason: str | None) -> NodeRow:
    detail = {} if reason is None else {'reason': reason}
    return NodeRow(
        path='r1',
        iteration=1,
        parent_path=None,
        level=0,
        member_count=5,
        eligible_count=5,
        membership_digest='d',
        snapshot_storage_key='k',
        step='done',
        decision=decision,
        class_name=None,
        answered=None,
        agree=None,
        bound=None,
        distribution={'votes': [], 'new_class_proposals': [], 'detail': detail},
        admitted_crops=None,
        admitted_requests=None,
        label_op_id=None,
        items_propagated=0,
        skipped={},
    )


@pytest.mark.parametrize(
    ('reason', 'kind'),
    [
        ('budget_crops', 'unvisited'),
        ('budget_requests', 'unvisited'),
        ('budget_tokens', 'unvisited'),
        ('budget_time', 'unvisited'),
        ('no_eligible_members', 'review_small'),
        ('no_split', 'review_leaf'),
        ('class_not_in_registry', 'stale'),
        ('drift', 'stale'),
    ],
)
def test_every_reason_the_loop_records_has_a_sentence(reason: str, kind: str) -> None:
    node = _node(_row(kind, reason))

    assert node.reason == reason
    assert node.reason_message
    assert node.reason_message != reason


def test_a_node_without_a_reason_has_no_sentence() -> None:
    assert _node(_row('pass', None)).reason_message is None
