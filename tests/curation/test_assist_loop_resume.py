"""Resume at every kill point: the loop dies just before or just after each kind of event
and a fresh process resumes it from the job directory and the items. No answered item is
asked again (an answer that never reached the log is read back from the item's ``assist``
slot), no write is lost or repeated, and the final tree and items are those of a run that
never died.
"""

from __future__ import annotations

import asyncio
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
from src.services.curation.assist_loop.store import LoopStore


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures('readable_crops')

LOOP = 'alp_20261028T000000Z_000001'
SERIAL = {'node_concurrency': 1, 'vlm_concurrency': 1}

KILL_POINTS = [
    ('node_open', 1, 'after'),
    ('node_open', 3, 'after'),
    ('reserve', 1, 'before'),
    ('reserve', 1, 'after'),
    ('reserve', 9, 'after'),
    ('answers', 1, 'before'),
    ('answers', 4, 'before'),
    ('answers', 4, 'after'),
    ('decision', 1, 'before'),
    ('decision', 1, 'after'),
    ('decision', 3, 'after'),
    ('children', 1, 'before'),
    ('children', 1, 'after'),
    ('propagate_begin', 1, 'after'),
    ('propagated', 1, 'before'),
    ('propagated', 1, 'after'),
    ('propagated', 3, 'before'),
]


def _populate(world: World) -> None:
    world.add_cluster(10001, {'car': 60, 'truck': 60, 'bus': 60, 'van': 60}, prefix='a')
    world.add_cluster(10002, {'car': 200}, prefix='b')


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(tmp_path, monkeypatch)
    _populate(w)
    return w


def _items(world: World) -> dict[str, Any]:
    return without_times(world.os.docs(ITEMS))


@pytest.mark.parametrize(('event', 'occurrence', 'when'), KILL_POINTS)
def test_a_loop_killed_at_any_event_resumes_to_the_same_end(
    world: World, tmp_path: Path, event: str, occurrence: int, when: str
) -> None:
    pristine = world.snapshot()
    reference = asyncio.run(
        drive(world, tmp_path / 'reference' / LOOP, FakeVlm(world.truth), **SERIAL)
    )
    expected_tree = decisions(reference.store)
    expected_items = _items(world)
    world.restore(pristine)

    vlm = FakeVlm(world.truth)
    loop_dir = tmp_path / 'killed' / LOOP
    store = LoopStore(loop_dir)
    switch = KillSwitch(event, occurrence, when)
    switch.arm(store, world.os)
    with pytest.raises(Killed):
        asyncio.run(drive(world, loop_dir, vlm, store=store, **SERIAL))
    switch.disarm()

    resumed = asyncio.run(drive(world, loop_dir, vlm, **SERIAL))

    assert resumed.outcome.status == 'completed'
    assert decisions(resumed.store) == expected_tree
    assert len(vlm.asked) == len(set(vlm.asked)), 'an answered item was asked again'
    assert _items(world) == expected_items
    for crop_id, doc in world.os.docs(ITEMS).items():
        writes = [
            h for h in doc.get('class_id_history') or [] if h['writer'] == f'assist_loop:{LOOP}'
        ]
        assert len(writes) <= 1, f'{crop_id} was propagated twice'


def test_a_killed_loop_recovers_an_unlogged_answer_from_the_item(
    world: World, tmp_path: Path
) -> None:
    vlm = FakeVlm(world.truth)
    loop_dir = tmp_path / LOOP
    store = LoopStore(loop_dir)
    switch = KillSwitch('answers', 1, 'before')
    switch.arm(store, world.os)
    with pytest.raises(Killed):
        asyncio.run(drive(world, loop_dir, vlm, store=store, **SERIAL))
    switch.disarm()
    in_flight = list(vlm.asked)

    resumed = asyncio.run(drive(world, loop_dir, vlm, **SERIAL))

    recovered = [
        e for e in resumed.store.events() if e['event'] == 'answers' and e.get('recovered')
    ]
    assert recovered, 'the reservation without its answers was looked up on the items'
    assert {c for e in recovered for c in e['answers']} == set(in_flight)
    assert all(vlm.asked.count(c) == 1 for c in in_flight)


def test_a_reservation_lost_in_a_crash_is_spent(world: World, tmp_path: Path) -> None:
    vlm = FakeVlm(world.truth)
    loop_dir = tmp_path / LOOP
    store = LoopStore(loop_dir)
    switch = KillSwitch('reserve', 1, 'after')
    switch.arm(store, world.os)
    with pytest.raises(Killed):
        asyncio.run(drive(world, loop_dir, vlm, store=store, **SERIAL))
    switch.disarm()

    resumed = asyncio.run(drive(world, loop_dir, vlm, **SERIAL))

    lost = next(e for e in resumed.store.events() if e['event'] == 'reserve')
    spent = resumed.budget.snapshot()['vlm']['crops_spent']
    assert spent == len(vlm.asked) + lost['crops'], 'the lost reservation counts as spent'
