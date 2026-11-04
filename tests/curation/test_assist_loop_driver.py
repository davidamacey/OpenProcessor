"""The loop end to end over a fake world: pass and propagate, split and recurse, the stop
rules (depth, node size, budget, time, cancel, pause), best-first scheduling, re-drawn
failures, per-level progress, and the precision/coverage/cost numbers of a fake VLM whose
accuracy is known.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from curation.assist_loop_fakes import ITEMS, FakeVlm, World, decisions, drive, make_world
from src.domain.loop_stats import required_sample
from src.services.curation.assist_loop.partition import PartitionScope


if TYPE_CHECKING:
    from pathlib import Path

    from src.services.curation.assist_loop.store import LoopStore

pytestmark = pytest.mark.usefixtures('readable_crops')


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return make_world(tmp_path, monkeypatch)


def _sources(world: World, prefix: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for crop_id, doc in world.os.docs(ITEMS).items():
        if crop_id.startswith(prefix):
            out[doc['class_source']] = out.get(doc['class_source'], 0) + 1
    return out


def test_a_homogeneous_root_passes_and_its_class_reaches_every_eligible_member(
    world: World, tmp_path: Path
) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    vlm = FakeVlm(world.truth)

    run = asyncio.run(drive(world, tmp_path / 'loop', vlm))

    k1 = required_sample(300, 0.9, 0.025, 0)
    assert run.outcome.status == 'completed'
    assert run.outcome.stop_reason == 'done'
    assert decisions(run.store) == {'r10001': 'pass'}
    assert len(vlm.asked) == k1
    assert _sources(world, 'a') == {'vlm': k1, 'assist_propagated': 300 - k1}
    assert all(d['class_name'] == 'car' for d in world.os.docs(ITEMS).values())
    state = run.store.job.read()
    assert state['status'] == 'completed'
    (level0,) = state['levels']
    assert level0['level'] == 0
    assert (level0['nodes'], level0['passed']) == (1, 1)
    assert level0['items_propagated'] == 300 - k1
    assert level0['items_direct'] == k1
    assert level0['crops_spent'] == k1
    assert state['totals']['items_propagated'] == 300 - k1
    assert state['budget']['vlm']['crops_spent'] == k1


def test_a_mixed_root_splits_and_its_homogeneous_children_pass(
    world: World, tmp_path: Path
) -> None:
    world.add_cluster(10001, {'car': 60, 'truck': 60, 'bus': 60, 'van': 60}, prefix='a')

    run = asyncio.run(drive(world, tmp_path / 'loop', FakeVlm(world.truth)))

    got = decisions(run.store)
    assert got['r10001'] == 'split'
    children = {p: k for p, k in got.items() if p != 'r10001'}
    assert len(children) == 4
    assert set(children.values()) == {'pass'}
    for crop_id, doc in world.os.docs(ITEMS).items():
        assert doc['class_name'] == world.truth[crop_id], crop_id
    levels = {row['level']: row for row in run.store.job.read()['levels']}
    assert levels[0]['split'] == 1
    assert levels[1]['passed'] == 4
    k_child = required_sample(60, 0.9, 0.025, 0)
    assert levels[1]['crops_spent'] < 4 * k_child, 'children reuse the answers of their parent'


def test_a_failing_root_at_the_depth_limit_is_left_for_review(world: World, tmp_path: Path) -> None:
    world.add_cluster(10001, {'car': 60, 'truck': 60}, prefix='a')

    run = asyncio.run(drive(world, tmp_path / 'loop', FakeVlm(world.truth), max_depth=0))

    assert decisions(run.store) == {'r10001': 'review_leaf'}
    assert 'assist_propagated' not in _sources(world, 'a')
    assert run.store.job.read()['totals']['clusters_left_for_review'] == 1


def test_a_failing_node_below_the_minimum_size_is_left_for_review(
    world: World, tmp_path: Path
) -> None:
    world.add_cluster(10001, {'car': 60, 'truck': 60}, prefix='a')

    run = asyncio.run(drive(world, tmp_path / 'loop', FakeVlm(world.truth), min_node_size=500))

    assert decisions(run.store) == {'r10001': 'review_small'}


def test_the_largest_node_is_scheduled_first_and_the_rest_go_unvisited(
    world: World, tmp_path: Path
) -> None:
    world.add_cluster(10001, {'truck': 100}, prefix='small')
    world.add_cluster(10002, {'car': 300}, prefix='big')
    cap = required_sample(300, 0.9, 0.025, 0) + 4

    run = asyncio.run(
        drive(
            world,
            tmp_path / 'loop',
            FakeVlm(world.truth),
            max_crops=cap,
            node_concurrency=8,
            scope=PartitionScope(cluster_ids=None, min_root_size=10),
        )
    )

    assert decisions(run.store) == {'r10002': 'pass', 'r10001': 'unvisited'}
    assert run.outcome.status == 'completed'
    assert run.outcome.stop_reason == 'budget_crops'
    assert run.budget.snapshot()['vlm']['crops_spent'] <= cap
    assert 'vlm' not in _sources(world, 'small')


def test_failed_answers_are_redrawn_once_and_spent(world: World, tmp_path: Path) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    vlm = FakeVlm(world.truth, fail_rate=0.2)

    run = asyncio.run(drive(world, tmp_path / 'loop', vlm, max_depth=0))

    k1 = required_sample(300, 0.9, 0.025, 0)
    assert len(vlm.asked) == len(set(vlm.asked)), 'a failed crop is not asked again'
    assert len(vlm.asked) > k1
    assert run.budget.snapshot()['vlm']['crops_spent'] == len(vlm.asked)
    (decision,) = [e for e in run.store.events() if e['event'] == 'decision']
    assert decision['sampled'] == len(vlm.asked)
    assert decision['answered'] == sum(1 for c in vlm.asked if vlm.answer_for(c) is not None)


def test_the_time_budget_stops_the_loop(world: World, tmp_path: Path) -> None:
    world.add_cluster(10001, {'truck': 100}, prefix='small')
    world.add_cluster(10002, {'car': 300}, prefix='big')
    now = [0.0]
    vlm = FakeVlm(world.truth)

    def tick(_ids: list[str]) -> None:
        now[0] += 10.0

    vlm.before_answer = tick

    run = asyncio.run(
        drive(world, tmp_path / 'loop', vlm, max_wall_s=50, clock=lambda: now[0], max_depth=0)
    )

    assert run.outcome.status == 'completed'
    assert run.outcome.stop_reason == 'budget_time'
    assert set(decisions(run.store).values()) == {'unvisited'}


def test_cancel_stops_new_work_and_a_resume_finishes_the_same_tree(
    world: World, tmp_path: Path
) -> None:
    world.add_cluster(10001, {'car': 60, 'truck': 60, 'bus': 60, 'van': 60}, prefix='a')
    pristine = world.snapshot()
    reference = asyncio.run(
        drive(world, tmp_path / 'reference', FakeVlm(world.truth), node_concurrency=8)
    )
    expected = decisions(reference.store)
    world.restore(pristine)
    again = world
    vlm = FakeVlm(again.truth)
    loop_dir = tmp_path / 'loop'

    def cancel_after_some(_ids: list[str]) -> None:
        if vlm.requests == 12:
            (loop_dir / 'cancel.flag').touch()

    vlm.before_answer = cancel_after_some
    first = asyncio.run(drive(again, loop_dir, vlm, node_concurrency=1))

    assert first.outcome.status == 'cancelled'
    assert decisions(first.store) != expected
    (loop_dir / 'cancel.flag').unlink()
    vlm.before_answer = None
    second = asyncio.run(drive(again, loop_dir, vlm, node_concurrency=1))

    assert second.outcome.status == 'completed'
    assert decisions(second.store) == expected
    assert len(vlm.asked) == len(set(vlm.asked))


def test_a_user_pause_stops_the_loop_as_paused(world: World, tmp_path: Path) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    world.add_cluster(10002, {'truck': 200}, prefix='b')
    loop_dir = tmp_path / 'loop'
    vlm = FakeVlm(world.truth)

    def pause(_ids: list[str]) -> None:
        loop_dir.mkdir(parents=True, exist_ok=True)
        (loop_dir / 'pause.flag').touch()

    vlm.before_answer = pause

    run = asyncio.run(drive(world, loop_dir, vlm, node_concurrency=1))

    assert run.outcome.status == 'paused'
    assert run.outcome.pause_reason == 'user'
    assert run.store.job.read()['pause_reason'] == 'user'


def test_a_fake_vlm_of_known_accuracy_gives_precise_cheap_labels(
    world: World, tmp_path: Path
) -> None:
    """Precision, coverage and cost of the loop over three roots: one large homogeneous root,
    one mixing two separable classes, and one the VLM is bad at."""
    world.add_cluster(10001, {'car': 1200}, prefix='a')
    world.add_cluster(10002, {'truck': 400, 'bus': 400}, prefix='b')
    world.add_cluster(10003, {'van': 300}, prefix='c')
    vlm = FakeVlm(
        world.truth,
        accuracy={'car': 0.995, 'truck': 0.995, 'bus': 0.995, 'van': 0.6},
        confusion={'van': 'car'},
        seed=1,
    )
    cap = 1000

    run = asyncio.run(
        drive(world, tmp_path / 'loop', vlm, max_crops=cap, theta=0.9, min_node_size=100)
    )

    docs = world.os.docs(ITEMS)
    propagated = [c for c, d in docs.items() if d['class_source'] == 'assist_propagated']
    direct = [c for c, d in docs.items() if d['class_source'] == 'vlm']
    correct = sum(1 for c in propagated if docs[c]['class_name'] == world.truth[c])
    spent = run.budget.snapshot()['vlm']['crops_spent']
    precision = correct / len(propagated)
    coverage = (len(propagated) + len(direct)) / len(docs)
    assert spent <= cap
    assert precision >= 0.95
    assert coverage >= 0.6
    assert spent / (len(propagated) + len(direct)) <= 0.15
    assert not any(c.startswith('c-') for c in propagated), 'the VLM is bad at this root'


def test_a_tight_budget_is_not_spent_on_nodes_it_cannot_finish(
    world: World, tmp_path: Path
) -> None:
    """Concurrent nodes must not each take a slice of a budget none of them can finish on:
    every crop spent belongs to a node that got a decision, the rest are left untouched."""
    for index in range(12):
        world.add_cluster(10001 + index, {'car': 20}, prefix=f'c{index}')
    need = required_sample(20, 0.9, 0.025, 1)
    cap = need * 4 + 3
    vlm = FakeVlm(world.truth, accuracy=0.93, seed=3)

    run = asyncio.run(
        drive(
            world,
            tmp_path / 'loop',
            vlm,
            max_crops=cap,
            node_concurrency=8,
            scope=PartitionScope(cluster_ids=None, min_root_size=10),
        )
    )

    final = decisions(run.store)
    spent_on: dict[str, int] = {}
    for event in run.store.events():
        if event['event'] == 'reserve':
            spent_on[event['path']] = spent_on.get(event['path'], 0) + event['crops']
    wasted = {p: c for p, c in spent_on.items() if final.get(p, 'unvisited') == 'unvisited'}
    assert wasted == {}
    assert sum(1 for kind in final.values() if kind != 'unvisited') >= 4
    assert run.outcome.stop_reason == 'budget_crops'
    assert run.budget.snapshot()['vlm']['crops_spent'] <= cap


def _dissenting_first_draw(world: World, tmp_path: Path, cluster: int, prefix: str) -> FakeVlm:
    """A VLM that contradicts exactly one crop of the first draw of ``prefix``'s root, so the
    node passes only after its dissent top-up. The draw is read off a probe run."""
    pristine = world.snapshot()
    probe = FakeVlm(world.truth)
    asyncio.run(
        drive(
            world,
            tmp_path / 'probe',
            probe,
            max_depth=0,
            scope=PartitionScope(cluster_ids=[cluster], min_root_size=10),
        )
    )
    world.restore(pristine)
    wrong = next(c for c in probe.asked if c.startswith(prefix))
    return FakeVlm(world.truth, answers={wrong: 'bus'})


def _crops_spent_on_unvisited(run_store: LoopStore) -> dict[str, int]:
    final = decisions(run_store)
    spent: dict[str, int] = {}
    for event in run_store.events():
        if event['event'] == 'reserve':
            spent[event['path']] = spent.get(event['path'], 0) + event['crops']
    return {p: c for p, c in spent.items() if final.get(p, 'unvisited') == 'unvisited'}


def test_a_node_that_needs_its_dissent_top_up_does_not_starve_the_next_one(
    world: World, tmp_path: Path
) -> None:
    """The first node of a loop is admitted on its first draw alone, but it may still need
    its top-up: the nodes admitted beside it must be counted against that full claim."""
    world.add_cluster(10001, {'car': 22}, prefix='a')
    world.add_cluster(10002, {'car': 20}, prefix='b')
    first_draw = required_sample(22, 0.9, 0.025, 0)
    first_full = required_sample(22, 0.9, 0.025, 1)
    second_first_draw = required_sample(20, 0.9, 0.025, 0)
    second_full = required_sample(20, 0.9, 0.025, 1)
    cap = first_draw + second_full
    assert cap < first_full + second_full
    assert cap < first_full + second_first_draw
    vlm = _dissenting_first_draw(world, tmp_path, 10001, 'a')

    run = asyncio.run(
        drive(
            world,
            tmp_path / 'loop',
            vlm,
            max_crops=cap,
            node_concurrency=2,
            max_depth=0,
            scope=PartitionScope(cluster_ids=None, min_root_size=10),
        )
    )

    assert decisions(run.store)['r10001'] == 'pass'
    assert _crops_spent_on_unvisited(run.store) == {}
    assert run.budget.snapshot()['vlm']['crops_spent'] <= cap


def test_a_request_limit_admits_nodes_like_the_crop_limit(world: World, tmp_path: Path) -> None:
    """Requests are a budget too: a node is not started beside another when the requests
    both may need are not there, and the reason recorded is the one that bound."""
    world.add_cluster(10001, {'car': 22}, prefix='a')
    world.add_cluster(10002, {'car': 20}, prefix='b')
    size = 4
    first_requests = -(-required_sample(22, 0.9, 0.025, 0) // size) + 2
    second_first_draw = -(-required_sample(20, 0.9, 0.025, 0) // size)
    max_requests = first_requests + second_first_draw - 1
    vlm = _dissenting_first_draw(world, tmp_path, 10001, 'a')

    run = asyncio.run(
        drive(
            world,
            tmp_path / 'loop',
            vlm,
            max_requests=max_requests,
            node_concurrency=2,
            max_depth=0,
            images_per_call=size,
            scope=PartitionScope(cluster_ids=None, min_root_size=10),
        )
    )

    assert decisions(run.store) == {'r10001': 'pass', 'r10002': 'unvisited'}
    assert _crops_spent_on_unvisited(run.store) == {}
    (reason,) = [
        e['reason']
        for e in run.store.events()
        if e['event'] == 'decision' and e['kind'] == 'unvisited'
    ]
    assert reason == 'budget_requests'
    assert run.outcome.stop_reason == 'budget_requests'


def test_a_resume_restarts_admitted_nodes_without_deciding_their_admission_again(
    world: World, tmp_path: Path
) -> None:
    world.add_cluster(10001, {'car': 22}, prefix='a')
    world.add_cluster(10002, {'car': 20}, prefix='b')
    loop_dir = tmp_path / 'loop'
    vlm = FakeVlm(world.truth)
    scope = PartitionScope(cluster_ids=None, min_root_size=10)

    def cancel(_ids: list[str]) -> None:
        loop_dir.mkdir(parents=True, exist_ok=True)
        (loop_dir / 'cancel.flag').touch()

    vlm.before_answer = cancel
    first = asyncio.run(drive(world, loop_dir, vlm, node_concurrency=2, max_depth=0, scope=scope))
    admitted = [e for e in first.store.events() if e['event'] == 'admit']
    (loop_dir / 'cancel.flag').unlink()
    vlm.before_answer = None

    second = asyncio.run(drive(world, loop_dir, vlm, node_concurrency=1, max_depth=0, scope=scope))

    assert {e['path'] for e in admitted} == {'r10001', 'r10002'}
    assert [e for e in second.store.events() if e['event'] == 'admit'] == admitted
    assert set(decisions(second.store).values()) == {'pass'}


class _OneRequestFails(FakeVlm):
    """Its first request raises; every other request is slow and records that it finished."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.finished: list[list[str]] = []

    async def label_or_propose_batch(self, crops: list[Any], *args: Any, **kwargs: Any) -> Any:
        if self.requests == 0:
            self.requests += 1
            msg = 'endpoint refused the request'
            raise RuntimeError(msg)
        self.requests += 1
        await asyncio.sleep(0.05)
        self.finished.append([c.img_id for c in crops])
        return await super().label_or_propose_batch(crops, *args, **kwargs)


def test_a_failing_request_cancels_the_sibling_requests_of_its_node(
    world: World, tmp_path: Path
) -> None:
    """The requests of one draw run together; when one raises, the rest are cancelled and
    awaited rather than left to finish (and spend) after the loop has already failed."""
    world.add_cluster(10001, {'car': 300}, prefix='a')
    vlm = _OneRequestFails(world.truth)

    async def scenario() -> Any:
        run = await drive(world, tmp_path / 'loop', vlm, node_concurrency=1, max_depth=0)
        await asyncio.sleep(0.3)
        return run

    run = asyncio.run(scenario())

    assert run.outcome.status == 'failed'
    assert vlm.finished == []


def test_a_failing_node_cancels_and_awaits_the_nodes_running_beside_it(
    world: World, tmp_path: Path
) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')
    world.add_cluster(10002, {'truck': 250}, prefix='b')
    vlm = _OneRequestFails(world.truth)

    async def scenario() -> Any:
        run = await drive(world, tmp_path / 'loop', vlm, node_concurrency=2, max_depth=0)
        await asyncio.sleep(0.3)
        return run

    run = asyncio.run(scenario())

    assert run.outcome.status == 'failed'
    assert vlm.finished == []
