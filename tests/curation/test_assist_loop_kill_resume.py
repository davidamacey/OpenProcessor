"""Kill the loop after (or just before) each step it takes, then resume it on the same tables:
no node is asked twice, no item is asked twice, and the final tree and items are those of a
run that never died. A crop is charged twice only as a forfeited, in-flight reservation.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from curation.assist_loop_fakes import (
    ITEMS,
    FakeVlm,
    Killed,
    KillSwitch,
    World,
    decisions,
    drive,
    make_world,
    without_times,
)


if TYPE_CHECKING:
    from curation.assist_loop_fakes import LoopEnv

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('readable_crops')]

SERIAL = {'node_concurrency': 1, 'vlm_concurrency': 1}

KILL_POINTS = [
    ('admit', 1, 'after'),
    ('select', 1, 'before'),
    ('select', 1, 'after'),
    ('select', 6, 'after'),
    ('record_answers', 1, 'before'),
    ('record_answers', 1, 'after'),
    ('record_answers', 4, 'before'),
    ('record_answers', 4, 'after'),
    ('decide', 1, 'before'),
    ('decide', 2, 'after'),
    ('split', 1, 'before'),
    ('split', 1, 'after'),
    ('gate_pass', 1, 'before'),
    ('gate_pass', 1, 'after'),
    ('begin_propagate', 1, 'after'),
    ('finish_propagate', 1, 'before'),
    ('finish_propagate', 1, 'after'),
    ('finish_propagate', 3, 'before'),
]


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(monkeypatch)
    w.add_cluster(10001, {'car': 60, 'truck': 60, 'bus': 60, 'van': 60}, prefix='a')
    w.add_cluster(10002, {'car': 200}, prefix='b')
    return w


def _items(world: World, *run_ids: Any) -> dict[str, Any]:
    """The items without timestamps and with every run id read as ``RUN`` (two runs of one
    world have different ids; what they wrote must be the same)."""
    text = json.dumps(without_times(world.os.docs(ITEMS)), sort_keys=True)
    for run_id in run_ids:
        text = text.replace(str(run_id), 'RUN')
    text = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 'ID', text)
    return json.loads(text)


@pytest.mark.parametrize(('event', 'occurrence', 'when'), KILL_POINTS)
def test_a_loop_killed_at_any_step_resumes_to_the_same_end(
    world: World, loop_env: LoopEnv, event: str, occurrence: int, when: str
) -> None:
    pristine = world.snapshot()
    reference = asyncio.run(drive(world, loop_env, FakeVlm(world.truth), **SERIAL))
    expected_items = _items(world, reference.ref.run_id)
    expected_tree = decisions(reference.record)
    expected_spent = reference.budget.snapshot()['vlm']['crops_spent']
    world.restore(pristine)

    ref = loop_env.new_run()
    vlm = FakeVlm(world.truth)
    switch = KillSwitch(event, occurrence, when)

    def arm(record: Any) -> None:
        switch.arm_record(record)
        switch.arm_os(world.os)

    with pytest.raises(Killed):
        asyncio.run(drive(world, loop_env, vlm, ref=ref, arm=arm, **SERIAL))
    assert switch.tripped
    switch.disarm()
    resumed = asyncio.run(drive(world, loop_env, vlm, ref=ref, **SERIAL))

    assert resumed.outcome.status == 'completed'
    assert decisions(resumed.record) == expected_tree
    assert _items(world, ref.run_id) == expected_items
    assert len(vlm.asked) == len(set(vlm.asked)), 'no item is asked twice'
    spent = resumed.budget.snapshot()['vlm']['crops_spent']
    in_flight = SERIAL['vlm_concurrency'] * 4
    assert expected_spent <= spent <= expected_spent + in_flight
