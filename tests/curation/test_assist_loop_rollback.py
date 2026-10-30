"""Rollback of what a loop propagated: only items still exactly as the loop left them are
restored, to the state before the loop wrote; anything a person or another writer changed
since is kept and counted. A dry run writes nothing; a second rollback is a no-op.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from curation.assist_loop_fakes import ITEMS, FakeVlm, World, drive, make_world
from src.domain.history import CLASS_STATE_FIELDS
from src.services.curation.assist_loop.rollback import rollback_loop


if TYPE_CHECKING:
    from pathlib import Path

    from curation.assist_loop_fakes import Run

pytestmark = pytest.mark.usefixtures('readable_crops')

LOOP = 'alp_20261028T000000Z_000004'


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(tmp_path, monkeypatch)
    w.add_cluster(10001, {'car': 200}, prefix='a')
    w.add_cluster(10002, {'truck': 150}, prefix='b')
    return w


def _class_state(doc: dict[str, Any]) -> dict[str, Any]:
    return {f: doc.get(f) for f in CLASS_STATE_FIELDS}


def _propagated(world: World) -> list[str]:
    return sorted(
        c for c, d in world.os.docs(ITEMS).items() if d['class_source'] == 'assist_propagated'
    )


def _loop(world: World, tmp_path: Path) -> Run:
    return asyncio.run(drive(world, tmp_path / LOOP, FakeVlm(world.truth), max_depth=0))


def _rollback(world: World, run: Run, **kw: Any) -> Any:
    return asyncio.run(rollback_loop(world.os, run.store, **kw))  # type: ignore[arg-type]


def test_a_dry_run_counts_and_writes_nothing(world: World, tmp_path: Path) -> None:
    run = _loop(world, tmp_path)
    after_loop = world.snapshot()

    result = _rollback(world, run, dry_run=True)

    assert result.dry_run
    assert result.restored == len(_propagated(world)) > 0
    assert result.kept_changed == 0
    assert world.snapshot() == after_loop


def test_rollback_restores_exactly_the_pre_loop_state(world: World, tmp_path: Path) -> None:
    before = world.snapshot()
    run = _loop(world, tmp_path)
    written = _propagated(world)
    direct = sorted(c for c, d in world.os.docs(ITEMS).items() if d['class_source'] == 'vlm')

    result = _rollback(world, run, dry_run=False)

    assert result.restored == len(written)
    for crop_id in written:
        assert _class_state(world.doc(crop_id)) == _class_state(before[crop_id]), crop_id
        assert (
            world.doc(crop_id)['class_id_history'][-1]['writer'] == f'assist_loop_rollback:{LOOP}'
        )
    for crop_id in direct:
        assert world.doc(crop_id)['class_source'] == 'vlm', 'a direct VLM answer is not propagated'
    (event,) = [e for e in run.store.events() if e['event'] == 'rollback']
    assert event['restored'] == len(written)


@pytest.mark.parametrize(
    'change',
    [
        {'class_name': 'bus', 'class_source': 'human', 'class_validated': True},
        {'class_name': 'car', 'class_source': 'vlm'},
        {'class_excluded': True},
        {'class_validated': True},
        {'test_holdout': True},
    ],
    ids=['human_relabel', 'vlm_reask', 'excluded', 'validated', 'holdout'],
)
def test_an_item_changed_after_the_loop_is_kept_and_counted(
    world: World, tmp_path: Path, change: dict[str, Any]
) -> None:
    run = _loop(world, tmp_path)
    touched = _propagated(world)[0]
    world.doc(touched).update(change)
    world.os._bump(ITEMS, touched)
    kept = dict(world.doc(touched))

    result = _rollback(world, run, dry_run=False)

    assert result.kept_changed == 1
    assert world.doc(touched) == kept


def test_a_human_history_entry_after_the_loop_counts_as_a_change(
    world: World, tmp_path: Path
) -> None:
    run = _loop(world, tmp_path)
    touched = _propagated(world)[0]
    doc = world.doc(touched)
    doc['class_id_history'] = [*doc['class_id_history'], {'writer': 'human:label_crop', 'at': 'x'}]
    kept = dict(doc)

    result = _rollback(world, run, dry_run=False)

    assert result.kept_changed == 1
    assert world.doc(touched) == kept


def test_a_second_rollback_is_a_no_op(world: World, tmp_path: Path) -> None:
    run = _loop(world, tmp_path)
    _rollback(world, run, dry_run=False)
    after_first = world.snapshot()

    again = _rollback(world, run, dry_run=False)

    assert (again.matched, again.restored, again.kept_changed) == (0, 0, 0)
    assert world.snapshot() == after_first


def test_rollback_can_be_limited_to_nodes(world: World, tmp_path: Path) -> None:
    run = _loop(world, tmp_path)
    b_before = {c: dict(d) for c, d in world.os.docs(ITEMS).items() if c.startswith('b-')}

    result = _rollback(world, run, node_paths=['r10001'], dry_run=False)

    assert result.restored > 0
    assert not any(c.startswith('a-') for c in _propagated(world))
    for crop_id, doc in b_before.items():
        assert world.doc(crop_id) == doc


def test_a_human_write_racing_the_rollback_wins(world: World, tmp_path: Path) -> None:
    run = _loop(world, tmp_path)
    racer = _propagated(world)[0]
    real_bulk = world.os.bulk
    raced = False

    async def bulk_with_race(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        nonlocal raced
        if not raced:
            raced = True
            world.doc(racer).update(class_name='bus', class_source='human', class_validated=True)
            world.os._bump(ITEMS, racer)
        return await real_bulk(body=body, **kw)

    world.os.bulk = bulk_with_race  # type: ignore[method-assign]

    result = _rollback(world, run, dry_run=False)

    assert result.kept_changed == 1
    assert world.doc(racer)['class_source'] == 'human'


def test_a_holdout_item_is_kept_and_counted_while_the_rest_are_restored(
    world: World, tmp_path: Path
) -> None:
    """The holdout freeze comes from the shared lock rule, so this pins that rollback
    honours it end to end (dry run and real), not a copy of the rule inside rollback."""
    run = _loop(world, tmp_path)
    written = _propagated(world)
    frozen = written[0]
    world.doc(frozen)['test_holdout'] = True
    world.os._bump(ITEMS, frozen)
    kept = dict(world.doc(frozen))

    dry = _rollback(world, run, dry_run=True)
    real = _rollback(world, run, dry_run=False)

    assert (dry.matched, dry.restored, dry.kept_changed) == (len(written), len(written) - 1, 1)
    assert (real.restored, real.kept_changed) == (len(written) - 1, 1)
    assert world.doc(frozen) == kept
    (event,) = [e for e in run.store.events() if e['event'] == 'rollback']
    assert event['kept_changed'] == 1
