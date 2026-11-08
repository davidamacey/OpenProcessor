"""What the loop publishes while it runs: decided nodes, budget thresholds, the audit, the
iteration, warnings; each a snapshot of its loop, under the event size cap."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from curation.assist_loop_fakes import FakeVlm, World, drive, make_world
from src.domain import assist_loop_events as ev
from src.services.curation.assist_loop.audit import AuditParams


if TYPE_CHECKING:
    from curation.assist_loop_fakes import LoopEnv

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('readable_crops')]


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(monkeypatch)
    w.add_cluster(10001, {'car': 60, 'truck': 60, 'bus': 60, 'van': 60}, prefix='a')
    w.add_cluster(10002, {'car': 300}, prefix='b')
    return w


def _of(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e for e in events if e['type'] == kind]


def test_a_run_announces_its_nodes_budgets_audit_and_iteration(
    world: World, loop_env: LoopEnv, captured_events: list[dict[str, Any]]
) -> None:
    audit = AuditParams(rate=0.1, min_per_batch=30, floor=0.85, batch_items=100)

    run = asyncio.run(
        drive(
            world,
            loop_env,
            FakeVlm(world.truth),
            audit=audit,
            judge=FakeVlm(world.truth),
            max_crops=150,
            on_budget='complete',
        )
    )

    loop_id = str(run.ref.run_id)
    loop_events = [e for e in captured_events if e['type'].startswith('assist_loop.')]
    assert loop_events
    assert all(e['loop_id'] == loop_id for e in loop_events)
    assert all(ev.size(e) <= ev.MAX_PAYLOAD_BYTES for e in loop_events)
    announced = {n['path'] for e in _of(loop_events, ev.NODES) for n in e['nodes']}
    assert announced == {p for p, n in run.record.tree.nodes.items() if n.step == 'done'}
    assert all(len(e['nodes']) <= ev.NODES_PER_EVENT for e in _of(loop_events, ev.NODES))
    budgets = _of(loop_events, ev.BUDGET)
    assert budgets, 'a pool crossed a threshold'
    assert {b['pool'] for b in budgets[0]['budgets']} == {'vlm', 'audit'}
    audits = _of(loop_events, ev.AUDIT)
    assert audits[-1]['status'] == 'ok'
    assert audits[-1]['floor'] == 0.85
    (closed,) = _of(loop_events, ev.ITERATION)
    assert (closed['ordinal'], closed['status']) == (1, 'closed')
    assert closed['counts']['levels']


def test_calls_without_usage_raise_one_warning(
    world: World, loop_env: LoopEnv, captured_events: list[dict[str, Any]]
) -> None:
    asyncio.run(drive(world, loop_env, FakeVlm(world.truth), max_depth=0))

    warnings = _of(captured_events, ev.WARNING)
    assert [w['code'] for w in warnings].count('usage_missing') == 1
    assert 'carried no usage' in json.dumps(warnings[0])
