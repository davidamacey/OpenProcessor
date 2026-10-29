"""The loop's record per level and in total, computed from its tree.

What ``state.json`` shows while a loop runs and after: per level how many nodes passed,
split or were left for review and why, how many items the level labeled by propagation
and directly (the sampled items the VLM answered), and the VLM crops it spent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.assist_loop.propagate import SKIP_REASONS


if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.services.curation.assist_loop.partition import MemberState
    from src.services.curation.assist_loop.store import LoopTree, NodeRecord

DECISIONS = ('pass', 'split', 'review_leaf', 'review_small', 'stale', 'unvisited')
LEFT_FOR_REVIEW = frozenset({'review_leaf', 'review_small', 'stale', 'unvisited'})


def _column(kind: str) -> str:
    """The level row's column of a decision (``pass`` is counted as ``passed``)."""
    return 'passed' if kind == 'pass' else kind


def _kind(node: NodeRecord) -> str | None:
    return node.decision['kind'] if node.decision else None


def _propagated(node: NodeRecord) -> int:
    return int(node.propagated['written']) if node.propagated else 0


def _direct(node: NodeRecord) -> int:
    return sum(1 for a in node.answers.values() if a.class_name)


def level_rows(tree: LoopTree) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for node in tree.nodes.values():
        row = rows.setdefault(
            node.level,
            {
                'level': node.level,
                'nodes': 0,
                **{_column(k): 0 for k in DECISIONS},
                'items_propagated': 0,
                'items_direct': 0,
                'crops_spent': 0,
            },
        )
        row['nodes'] += 1
        kind = _kind(node)
        if kind is not None:
            row[_column(kind)] += 1
        row['items_propagated'] += _propagated(node)
        row['items_direct'] += _direct(node)
        row['crops_spent'] += node.crops
    return [rows[level] for level in sorted(rows)]


def totals(tree: LoopTree, members: Iterable[MemberState]) -> dict[str, Any]:
    left = [n for n in tree.nodes.values() if _kind(n) in LEFT_FOR_REVIEW]
    states: dict[str, int] = {'locked': 0, 'holdout': 0, 'excluded': 0}
    for member in members:
        if member.state in states:
            states[member.state] += 1
    skipped = {**dict.fromkeys(SKIP_REASONS, 0), **states}
    for node in tree.nodes.values():
        for key, value in ((node.propagated or {}).get('skipped') or {}).items():
            if key in skipped:
                skipped[key] += int(value)
    return {
        'items_propagated': sum(_propagated(n) for n in tree.nodes.values()),
        'items_direct': sum(_direct(n) for n in tree.nodes.values()),
        'clusters_left_for_review': len(left),
        'items_left_for_review': sum(n.n for n in left),
        'skipped': skipped,
    }


__all__ = ['LEFT_FOR_REVIEW', 'level_rows', 'totals']
