"""Three pause cycles complete the run (a pause is not a failure, and nothing is asked twice),
and slices of ``slice_s`` hand the loop over at node boundaries to the same end."""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from curation.assist_loop_fakes import (
    ITEMS,
    FakeVlm,
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


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(monkeypatch)
    w.add_cluster(10001, {'car': 60, 'truck': 60, 'bus': 60, 'van': 60}, prefix='a')
    w.add_cluster(10002, {'car': 200}, prefix='b')
    w.add_cluster(10003, {'truck': 150}, prefix='c')
    return w


def _items(world: World, *run_ids: Any) -> dict[str, Any]:
    """The items without timestamps and with every run id read as ``RUN`` (two runs of one
    world have different ids; what they wrote must be the same)."""
    text = json.dumps(without_times(world.os.docs(ITEMS)), sort_keys=True)
    for run_id in run_ids:
        text = text.replace(str(run_id), 'RUN')
    text = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 'ID', text)
    return json.loads(text)


def test_three_pause_cycles_complete_the_run_to_the_same_end(
    world: World, loop_env: LoopEnv
) -> None:
    pristine = world.snapshot()
    reference = asyncio.run(drive(world, loop_env, FakeVlm(world.truth), **SERIAL))
    expected, expected_items = decisions(reference.record), _items(world, reference.ref.run_id)
    world.restore(pristine)
    ref = loop_env.new_run()
    vlm = FakeVlm(world.truth)
    cycles = 0

    for pause_at in (4, 10, 18):
        ref.pause = False

        def pause_when_asked(_ids: list[str], pause_at: int = pause_at) -> None:
            if vlm.requests >= pause_at:
                ref.pause = True

        vlm.before_answer = pause_when_asked
        run = asyncio.run(drive(world, loop_env, vlm, ref=ref, **SERIAL))
        assert run.outcome.status == 'paused'
        cycles += 1
    ref.pause = False
    vlm.before_answer = None
    final = asyncio.run(drive(world, loop_env, vlm, ref=ref, **SERIAL))

    assert cycles == 3
    assert final.outcome.status == 'completed'
    assert decisions(final.record) == expected
    assert _items(world, ref.run_id) == expected_items
    assert len(vlm.asked) == len(set(vlm.asked))


def test_slices_hand_the_loop_over_at_node_boundaries_to_the_same_end(
    world: World, loop_env: LoopEnv
) -> None:
    pristine = world.snapshot()
    reference = asyncio.run(drive(world, loop_env, FakeVlm(world.truth), **SERIAL))
    expected, expected_items = decisions(reference.record), _items(world, reference.ref.run_id)
    world.restore(pristine)
    ref = loop_env.new_run()
    vlm = FakeVlm(world.truth)
    slices = 0

    while True:
        start = vlm.requests
        run = asyncio.run(
            drive(
                world,
                loop_env,
                vlm,
                ref=ref,
                slice_due=lambda start=start: vlm.requests > start,
                **SERIAL,
            )
        )
        slices += 1
        assert slices < 40
        if run.outcome.status == 'completed':
            break
        assert run.outcome.status == 'yield'

    assert slices > 3
    assert decisions(run.record) == expected
    assert _items(world, ref.run_id) == expected_items
    assert len(vlm.asked) == len(set(vlm.asked))
