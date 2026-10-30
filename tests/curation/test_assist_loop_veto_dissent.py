"""Ground truth the node already holds overrules the VLM's agreement, and a sampled member that
disagreed keeps its own answer instead of the node's class.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from curation.assist_loop_fakes import ITEMS, FakeVlm, World, decisions, drive, make_world
from src.domain.loop_stats import required_sample
from src.domain.loop_tree import child_sample


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures('readable_crops')

ROOT = 10001


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(tmp_path, monkeypatch)
    w.add_cluster(ROOT, {'car': 200}, prefix='a')
    return w


def _human(world: World, crop_id: str, name: str) -> None:
    world.add_item(
        crop_id,
        ROOT,
        name,
        class_id=world.class_id(name),
        class_name=name,
        class_source='human',
        label_source='human',
        class_validated=True,
    )


def test_validated_members_disagreeing_block_the_pass(world: World, tmp_path: Path) -> None:
    for i in range(3):
        _human(world, f'h{i}', 'truck')  # the VLM confuses trucks for cars in this root

    run = asyncio.run(drive(world, tmp_path / 'loop', FakeVlm(world.truth), max_depth=0))

    assert decisions(run.store) == {'r10001': 'review_leaf'}
    (decision,) = [e for e in run.store.events() if e['event'] == 'decision']
    assert decision['vetoed'] is True
    assert decision['class_name'] == 'car'
    assert not any(d['class_source'] == 'assist_propagated' for d in world.os.docs(ITEMS).values())


def test_fewer_than_three_validated_members_do_not_veto(world: World, tmp_path: Path) -> None:
    for i in range(2):
        _human(world, f'h{i}', 'truck')

    run = asyncio.run(drive(world, tmp_path / 'loop', FakeVlm(world.truth), max_depth=0))

    assert decisions(run.store) == {'r10001': 'pass'}


def test_validated_members_agreeing_do_not_veto(world: World, tmp_path: Path) -> None:
    for i in range(5):
        _human(world, f'h{i}', 'car')

    run = asyncio.run(drive(world, tmp_path / 'loop', FakeVlm(world.truth), max_depth=0))

    assert decisions(run.store) == {'r10001': 'pass'}


def test_a_dissenting_sampled_member_keeps_its_own_vlm_answer(world: World, tmp_path: Path) -> None:
    eligible = sorted(world.truth)
    first_draw = child_sample(
        [], eligible, required_sample(200, 0.9, 0.025, 0), seed=0, path='r10001'
    )
    dissenter, agreer = first_draw.draw[:2]
    vlm = FakeVlm(world.truth, answers={dissenter: 'van'})

    run = asyncio.run(drive(world, tmp_path / 'loop', vlm, max_depth=0))

    assert decisions(run.store) == {'r10001': 'pass'}
    doc = world.doc(dissenter)
    assert (doc['class_name'], doc['class_source']) == ('van', 'vlm')
    agreeing = world.doc(agreer)
    assert (agreeing['class_name'], agreeing['class_source']) == ('car', 'vlm')
    (propagated,) = [e for e in run.store.events() if e['event'] == 'propagated']
    assert dissenter not in propagated['written_ids']
    assert propagated['skipped']['dissent'] == 1
