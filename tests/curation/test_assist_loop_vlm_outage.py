"""A VLM outage must stop the loop, not be read as a node that fails the gate (#286).

A node whose sampled crops all got a failed request (endpoint down, unknown model) has no
evidence either way: it is left open and the slice stops with ``VlmFailedError`` (the handler
pauses the job as ``vlm_unavailable``), spending nothing more and writing nothing. A VLM that answered but named nothing
usable is a legitimate answer outcome and keeps deciding the gate.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from curation.assist_loop_fakes import ITEMS, FakeVlm, World, decisions, drive, make_world, reload
from src.domain.loop_stats import required_sample
from src.services.curation.assist_loop.vlm_failure import VlmFailedError


if TYPE_CHECKING:
    from curation.assist_loop_fakes import LoopEnv

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('readable_crops')]

K1 = required_sample(300, 0.9, 0.025, 0)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    return make_world(monkeypatch)


def _classes(world: World) -> dict[str, object]:
    return {k: v['class_name'] for k, v in world.os.docs(ITEMS).items()}


def test_every_call_failing_stops_the_slice_with_the_error_and_spends_one_node(
    world: World, loop_env: LoopEnv
) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    world.add_cluster(10002, {'truck': 200}, prefix='b')
    vlm = FakeVlm(world.truth, fail_rate=1.0)
    ref = loop_env.new_run(max_crops=1000)

    with pytest.raises(VlmFailedError):
        asyncio.run(drive(world, loop_env, vlm, ref=ref, node_concurrency=1))

    record = asyncio.run(reload(loop_env, ref))
    assert len(vlm.asked) <= K1
    assert sum(n.crops for n in record.tree.nodes.values()) == len(vlm.asked)
    assert decisions(record) == {}, 'no node is split, reviewed or passed'
    assert all(n.step != 'propagating' and not n.children for n in record.tree.nodes.values())
    assert set(_classes(world).values()) == {None}


def test_a_failed_node_is_left_open_and_a_resume_asks_it_again(
    world: World, loop_env: LoopEnv
) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    ref = loop_env.new_run()
    with pytest.raises(VlmFailedError):
        asyncio.run(
            drive(world, loop_env, FakeVlm(world.truth, fail_rate=1.0), ref=ref, max_depth=0)
        )
    record = asyncio.run(reload(loop_env, ref))
    assert record.tree.node('r10001').failed == set(), 'an outage is forgotten, not a verdict'

    vlm = FakeVlm(world.truth)
    second = asyncio.run(drive(world, loop_env, vlm, ref=ref, max_depth=0))

    assert second.outcome.status == 'completed'
    assert decisions(second.record) == {'r10001': 'pass'}
    assert len(vlm.asked) == K1


def test_some_answers_among_failures_still_reach_the_gate(world: World, loop_env: LoopEnv) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    run = asyncio.run(drive(world, loop_env, FakeVlm(world.truth, fail_rate=0.3), max_depth=0))

    assert run.outcome.status == 'completed'
    assert decisions(run.record) == {'r10001': 'pass'}


def test_answers_that_name_nothing_usable_keep_deciding_the_gate(
    world: World, loop_env: LoopEnv
) -> None:
    ids = world.add_cluster(10001, {'car': 300}, prefix='a')
    vlm = FakeVlm(world.truth, answers=dict.fromkeys(ids, ''))

    run = asyncio.run(drive(world, loop_env, vlm, max_depth=0))

    assert run.outcome.status == 'completed'
    assert run.outcome.stop_reason == 'done'
    assert decisions(run.record) == {'r10001': 'review_leaf'}
    assert set(_classes(world).values()) == {None}


def test_a_top_up_whose_calls_all_fail_stops_instead_of_deciding(
    world: World, loop_env: LoopEnv
) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    vlm = FakeVlm(world.truth)

    def outage_after_first_stage(ids: list[str]) -> None:
        if not vlm.answers:
            vlm.answers[ids[0]] = 'truck'  # one dissent: the gate asks for a top-up
        if len(vlm.asked) > K1:
            vlm.fail_rate = 1.0

    vlm.before_answer = outage_after_first_stage

    ref = loop_env.new_run()
    with pytest.raises(VlmFailedError):
        asyncio.run(drive(world, loop_env, vlm, ref=ref, max_depth=0, node_concurrency=1))

    assert len(vlm.asked) > K1, 'the top-up was attempted'
    assert decisions(asyncio.run(reload(loop_env, ref))) == {}
