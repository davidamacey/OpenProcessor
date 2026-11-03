"""A node's ``reason`` code is served with a sentence for it, so the client renders what the
backend decided instead of keeping its own wording."""

from __future__ import annotations

import pytest

from src.services.curation.assist_loop.store import NodeRecord
from src.services.curation.assist_loop.views import _node


@pytest.mark.parametrize(
    ('reason', 'kind'),
    [
        ('budget_crops', 'unvisited'),
        ('budget_requests', 'unvisited'),
        ('budget_time', 'unvisited'),
        ('no_eligible_members', 'review_small'),
        ('no_split', 'review_leaf'),
        ('class_not_in_registry', 'stale'),
        ('drift', 'stale'),
    ],
)
def test_every_reason_the_loop_records_has_a_sentence(reason: str, kind: str) -> None:
    record = NodeRecord(path='r1', parent=None, level=0, n=5, digest='d')
    record.decision = {'kind': kind, 'reason': reason}

    node = _node(record)

    assert node.reason == reason
    assert node.reason_message
    assert node.reason_message != reason


def test_a_node_without_a_reason_has_no_sentence() -> None:
    record = NodeRecord(path='r1', parent=None, level=0, n=5, digest='d')
    record.decision = {'kind': 'pass'}

    assert _node(record).reason_message is None
