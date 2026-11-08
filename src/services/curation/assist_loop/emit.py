"""Publish the loop's events: what a client watching the project's feed sees while a loop runs.

The events are snapshots for a screen, not the record (the tables are): decided nodes at most
once a second in batches of 40, a pool's budget when it crosses 50, 80, 95 or 100 percent of a
limit (and after an edit), the audit after each batch, the iteration when it opens or closes, and
a warning at most once a minute per code. A failed publish is the caller's error: an event
that cannot be written means the database is not there either.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.domain import assist_loop_events as ev
from src.services.events import publish as events


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.services.curation.assist_loop.audit import Auditor
    from src.services.curation.assist_loop.budget import LoopBudget
    from src.services.curation.assist_loop.tree import LoopTree, NodeRecord


def node_event(node: NodeRecord) -> dict[str, Any]:
    decision = node.decision or {}
    return {
        'path': node.path,
        'level': node.level,
        'decision': decision.get('kind'),
        'class_name': decision.get('class_name'),
        'answered': decision.get('answered'),
        'agree': decision.get('agree'),
        'bound': decision.get('bound'),
        'eligible': node.n,
        'items_propagated': (node.propagated or {}).get('written', 0),
    }


class LoopEvents:
    """The events of one slice. ``clock`` is injectable for the throttles."""

    def __init__(self, loop_id: str, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.loop_id = loop_id
        self._clock = clock
        self._announced: set[str] = set()
        self._nodes_at = float('-inf')
        self._percent: dict[tuple[str, str], int] = {}
        self._warned: dict[str, float] = {}
        self._usage_missing = 0

    async def nodes(self, tree: LoopTree, *, force: bool = False) -> None:
        """Decided nodes not announced yet, in batches, at most once per second."""
        fresh = sorted(
            (n for n in tree.nodes.values() if n.step == 'done' and n.path not in self._announced),
            key=lambda n: (n.level, n.path),
        )
        if not fresh or (not force and self._clock() - self._nodes_at < ev.NODE_MIN_INTERVAL_S):
            return
        batches = [
            fresh[i : i + ev.NODES_PER_EVENT] for i in range(0, len(fresh), ev.NODES_PER_EVENT)
        ]
        await events.apublish_many(
            [
                (ev.NODES, ev.nodes_payload(self.loop_id, [node_event(n) for n in b]))
                for b in batches
            ]
        )
        self._announced.update(n.path for n in fresh)
        self._nodes_at = self._clock()

    async def budget(self, budget: LoopBudget, *, force: bool = False) -> None:
        """The budgets, when any pool unit crossed a threshold (or ``force``: after an edit)."""
        rows = budget.rows()
        hit = force
        for row in rows:
            now = ev.percent(row.spent, row.reserved, row.limit_value)
            key = (row.pool, row.unit)
            if ev.crossed(self._percent.get(key), now) is not None:
                hit = True
            self._percent[key] = now or 0
        if hit:
            await events.apublish(
                ev.BUDGET,
                ev.budget_payload(
                    self.loop_id,
                    [
                        {
                            'pool': r.pool,
                            'unit': r.unit,
                            'limit_value': r.limit_value,
                            'reserved': r.reserved,
                            'spent': r.spent,
                        }
                        for r in rows
                    ],
                ),
            )

    async def audit(self, auditor: Auditor, tree: LoopTree) -> None:
        await events.apublish(
            ev.AUDIT, ev.audit_payload(self.loop_id, auditor.summary(tree), auditor.params.floor)
        )

    async def iteration(self, ordinal: int, status: str, counts: dict[str, Any]) -> None:
        await events.apublish(
            ev.ITERATION, ev.iteration_payload(self.loop_id, ordinal, status, counts)
        )

    async def warning(self, code: str, detail: str) -> None:
        """At most once per :data:`~src.domain.assist_loop_events.WARNING_MIN_INTERVAL_S` per code."""
        now = self._clock()
        if now - self._warned.get(code, float('-inf')) < ev.WARNING_MIN_INTERVAL_S:
            return
        self._warned[code] = now
        await events.apublish(ev.WARNING, ev.warning_payload(self.loop_id, code, detail))

    async def warnings(self, tree: LoopTree, usage_missing: int) -> None:
        """Conditions a person should see: calls without usage, nodes gone stale."""
        if usage_missing > self._usage_missing:
            self._usage_missing = usage_missing
            await self.warning(
                'usage_missing',
                f'{usage_missing} answered calls carried no usage; charged at the reservation',
            )
        stale = [n for n in tree.nodes.values() if (n.decision or {}).get('kind') == 'stale']
        if stale:
            drifted = sum(1 for n in stale if (n.decision or {}).get('reason') == 'drift')
            await self.warning('stale_nodes', f'{len(stale)} nodes stale, {drifted} by drift')
            if drifted:
                await self.warning('drift', f'{drifted} nodes changed after the snapshot')


__all__ = ['LoopEvents', 'node_event']
