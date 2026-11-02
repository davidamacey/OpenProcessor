"""A VLM outage must stop the loop, not be read as a node that fails the gate (#286).

A node whose sampled crops all got a failed request (endpoint down, unknown model) has no
evidence either way: it is left open and the run stops ``failed`` / ``vlm_failed`` with the
error, spending nothing more and writing nothing. A VLM that answered but named nothing
usable is a legitimate answer outcome and keeps deciding the gate.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from curation.assist_loop_fakes import ITEMS, FakeVlm, World, decisions, drive, make_world
from src.domain.loop_stats import required_sample
from src.services.curation.assist_loop.store import replay


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures('readable_crops')

K1 = required_sample(300, 0.9, 0.025, 0)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return make_world(tmp_path, monkeypatch)


def _classes(world: World) -> dict[str, object]:
    return {k: v['class_name'] for k, v in world.os.docs(ITEMS).items()}


def test_every_call_failing_stops_the_run_with_the_error_and_spends_one_node(
    world: World, tmp_path: Path
) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    world.add_cluster(10002, {'truck': 200}, prefix='b')
    vlm = FakeVlm(world.truth, fail_rate=1.0)

    run = asyncio.run(drive(world, tmp_path / 'loop', vlm, node_concurrency=1, max_crops=1000))

    assert run.outcome.status == 'failed'
    assert run.outcome.stop_reason == 'vlm_failed'
    assert run.outcome.error
    state = run.store.job.read()
    assert (state['status'], state['stop_reason']) == ('failed', 'vlm_failed')
    assert state['error']
    assert len(vlm.asked) <= K1
    assert run.budget.snapshot()['vlm']['crops_spent'] == len(vlm.asked)
    assert decisions(run.store) == {}, 'no node is split, reviewed or passed'
    assert not any(e['event'] in ('children', 'propagate_begin') for e in run.store.events())
    assert set(_classes(world).values()) == {None}


def test_a_failed_node_is_left_open_and_a_resume_asks_it_again(
    world: World, tmp_path: Path
) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    first = asyncio.run(
        drive(world, tmp_path / 'loop', FakeVlm(world.truth, fail_rate=1.0), max_depth=0)
    )
    assert first.outcome.stop_reason == 'vlm_failed'
    assert replay(first.store.events()).nodes['r10001'].failed == set()

    vlm = FakeVlm(world.truth)
    second = asyncio.run(drive(world, tmp_path / 'loop', vlm, max_depth=0))

    assert second.outcome.status == 'completed'
    assert decisions(second.store) == {'r10001': 'pass'}
    assert len(vlm.asked) == K1


def test_some_answers_among_failures_still_reach_the_gate(world: World, tmp_path: Path) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    run = asyncio.run(
        drive(world, tmp_path / 'loop', FakeVlm(world.truth, fail_rate=0.3), max_depth=0)
    )

    assert run.outcome.status == 'completed'
    assert decisions(run.store) == {'r10001': 'pass'}


def test_answers_that_name_nothing_usable_keep_deciding_the_gate(
    world: World, tmp_path: Path
) -> None:
    ids = world.add_cluster(10001, {'car': 300}, prefix='a')
    vlm = FakeVlm(world.truth, answers=dict.fromkeys(ids, ''))

    run = asyncio.run(drive(world, tmp_path / 'loop', vlm, max_depth=0))

    assert run.outcome.status == 'completed'
    assert run.outcome.stop_reason == 'done'
    assert decisions(run.store) == {'r10001': 'review_leaf'}
    assert set(_classes(world).values()) == {None}


def test_a_top_up_whose_calls_all_fail_stops_instead_of_deciding(
    world: World, tmp_path: Path
) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    vlm = FakeVlm(world.truth)

    def outage_after_first_stage(ids: list[str]) -> None:
        if not vlm.answers:
            vlm.answers[ids[0]] = 'truck'  # one dissent: the gate asks for a top-up
        if len(vlm.asked) > K1:
            vlm.fail_rate = 1.0

    vlm.before_answer = outage_after_first_stage

    run = asyncio.run(drive(world, tmp_path / 'loop', vlm, max_depth=0, node_concurrency=1))

    assert len(vlm.asked) > K1, 'the top-up was attempted'
    assert run.outcome.status == 'failed'
    assert run.outcome.stop_reason == 'vlm_failed'
    assert decisions(run.store) == {}
