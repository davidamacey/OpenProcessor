"""The events an assist loop publishes, as pure payload builders (research auto-label plan 5.4).

Each payload is a snapshot (no deltas), carries ``loop_id`` and stays under the 7,000 byte cap of
the ``events`` table: a list is cut from its tail until it fits, and a payload that cannot fit
even empty is a bug (:class:`PayloadTooLargeError`). The loop tables, not these rows, are the
record: a client that missed an event re-reads the loop.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Final


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

NODES: Final = 'assist_loop.nodes'
BUDGET: Final = 'assist_loop.budget'
AUDIT: Final = 'assist_loop.audit'
ITERATION: Final = 'assist_loop.iteration'
WARNING: Final = 'assist_loop.warning'
TYPES: Final = (NODES, BUDGET, AUDIT, ITERATION, WARNING)

MAX_PAYLOAD_BYTES: Final = 7000
NODES_PER_EVENT: Final = 40
BUDGET_THRESHOLDS: Final = (50, 80, 95, 100)
"""Percent of a limit at which a pool's budget is published."""
WARNING_CODES: Final = ('vlm_error_rate', 'usage_missing', 'stale_nodes', 'drift')
WARNING_MIN_INTERVAL_S: Final = 60.0
NODE_MIN_INTERVAL_S: Final = 1.0
_NODE_FIELDS: Final = (
    'path',
    'level',
    'decision',
    'class_name',
    'answered',
    'agree',
    'bound',
    'eligible',
    'items_propagated',
)


class PayloadTooLargeError(ValueError):
    """A payload over the cap with nothing left to cut."""


def size(payload: Mapping[str, Any]) -> int:
    return len(json.dumps(payload, separators=(',', ':'), default=str).encode('utf-8'))


def fit(payload: dict[str, Any], list_key: str) -> dict[str, Any]:
    """``payload`` with ``list_key`` cut from its tail until it is under the cap."""
    items = list(payload[list_key])
    while size({**payload, list_key: items}) > MAX_PAYLOAD_BYTES:
        if not items:
            raise PayloadTooLargeError(f'{payload.get("loop_id")}: {list_key} cannot fit')
        items.pop()
    return {**payload, list_key: items}


def node_entry(node: Mapping[str, Any]) -> dict[str, Any]:
    return {k: node.get(k) for k in _NODE_FIELDS}


def nodes_payload(loop_id: str, nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Up to :data:`NODES_PER_EVENT` decided nodes."""
    entries = [node_entry(n) for n in nodes[:NODES_PER_EVENT]]
    return fit({'loop_id': loop_id, 'nodes': entries}, 'nodes')


def budget_payload(loop_id: str, budgets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``budgets``: ``{pool, unit, limit_value, reserved, spent}`` per row."""
    entries = [
        {k: row[k] for k in ('pool', 'unit', 'limit_value', 'reserved', 'spent')} for row in budgets
    ]
    return fit({'loop_id': loop_id, 'budgets': entries}, 'budgets')


def audit_payload(loop_id: str, summary: Mapping[str, Any], floor: float) -> dict[str, Any]:
    return {
        'loop_id': loop_id,
        'sampled': summary.get('sampled', 0),
        'agreed': summary.get('agreed', 0),
        'lower_bound': summary.get('lower_bound'),
        'floor': floor,
        'status': summary.get('status', 'pending'),
    }


def iteration_payload(
    loop_id: str, ordinal: int, status: str, counts: Mapping[str, Any]
) -> dict[str, Any]:
    """The iteration's per-level rows and totals; levels are cut from the deepest first."""
    levels = list(counts.get('levels') or [])
    totals = dict(counts.get('totals') or {})
    while True:
        payload = {
            'loop_id': loop_id,
            'ordinal': ordinal,
            'status': status,
            'counts': {'levels': levels, 'totals': totals},
        }
        if size(payload) <= MAX_PAYLOAD_BYTES:
            return payload
        if not levels:
            raise PayloadTooLargeError(f'{loop_id}: iteration counts cannot fit')
        levels = levels[:-1]


def warning_payload(loop_id: str, code: str, detail: str) -> dict[str, Any]:
    if code not in WARNING_CODES:
        raise ValueError(f'not a warning code: {code}')
    return {'loop_id': loop_id, 'code': code, 'detail': detail[:500]}


def percent(spent: int, reserved: int, limit: int | None) -> int | None:
    """How much of a limit is used (spent plus reserved); ``None`` for an uncapped unit."""
    if limit is None or limit <= 0:
        return None
    return min(100, (100 * (spent + reserved)) // limit)


def crossed(before: int | None, after: int | None) -> int | None:
    """The highest threshold passed going from ``before`` to ``after`` percent, if any."""
    if after is None:
        return None
    reached = [t for t in BUDGET_THRESHOLDS if after >= t > (before or 0)]
    return max(reached) if reached else None


__all__ = [
    'AUDIT',
    'BUDGET',
    'BUDGET_THRESHOLDS',
    'ITERATION',
    'MAX_PAYLOAD_BYTES',
    'NODES',
    'NODES_PER_EVENT',
    'NODE_MIN_INTERVAL_S',
    'TYPES',
    'WARNING',
    'WARNING_CODES',
    'WARNING_MIN_INTERVAL_S',
    'PayloadTooLargeError',
    'audit_payload',
    'budget_payload',
    'crossed',
    'fit',
    'iteration_payload',
    'node_entry',
    'nodes_payload',
    'percent',
    'size',
    'warning_payload',
]
