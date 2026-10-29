"""The loop never writes a class a person or an import owns, a validated class, a holdout
item or an excluded item; one case per item class. An unvalidated import and a VLM-labeled
item are machine labels and are written. Boxes are never touched.
"""

from __future__ import annotations

import asyncio
import copy
from typing import TYPE_CHECKING, Any

import pytest

from curation.assist_loop_fakes import BOXES, ITEMS, FakeVlm, World, drive, make_world
from src.domain.loop_stats import required_sample
from src.domain.loop_tree import child_sample, root_path
from src.services.curation.assist_loop.partition import ClusterFieldPartition
from src.services.curation.assist_loop.propagate import propagate_node
from src.services.curation.class_label import ItemLabel
from src.services.curation.class_write_guard import class_state_token


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures('readable_crops')

ROOT = 10001

NEVER_WRITTEN: dict[str, dict[str, Any]] = {
    'human': {'class_source': 'human', 'class_validated': True},
    'human_move': {'class_source': 'human_move', 'class_validated': True},
    'vlm_human_confirmed': {'class_source': 'vlm_human_confirmed', 'class_validated': True},
    'validated_import': {
        'class_source': 'external_label',
        'label_source': 'import',
        'class_validated': True,
    },
    'auto_promoted': {
        'class_source': 'cluster_majority_agreement',
        'class_validated': True,
    },
    'holdout': {'class_source': 'vlm', 'test_holdout': True},
    'excluded': {'class_source': 'vlm', 'class_excluded': True},
}
WRITTEN: dict[str, dict[str, Any]] = {
    'suggested_import': {
        'class_source': 'external_label',
        'label_source': 'import',
        'class_validated': False,
    },
    'vlm_labeled': {'class_source': 'vlm', 'label_source': 'vlm', 'class_validated': False},
}


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(tmp_path, monkeypatch)
    w.add_cluster(ROOT, {'car': 200}, prefix='free')
    truck = w.class_id('truck')
    for name, fields in {**NEVER_WRITTEN, **WRITTEN}.items():
        w.add_item(name, ROOT, 'car', class_id=truck, class_name='truck', **fields)
    return w


def test_no_locked_validated_holdout_or_excluded_item_is_ever_written(
    world: World, tmp_path: Path
) -> None:
    before = copy.deepcopy(world.os.docs(ITEMS))
    vlm = FakeVlm(world.truth)

    run = asyncio.run(drive(world, tmp_path / 'loop', vlm))

    assert run.outcome.status == 'completed'
    assert any(
        world.doc(f'free-car-{i:04d}')['class_source'] == 'assist_propagated' for i in range(200)
    )
    for name in NEVER_WRITTEN:
        assert world.doc(name) == before[name], f'{name} was written'
        assert name not in vlm.asked, f'{name} was sent to the VLM'


def test_an_unvalidated_import_and_a_vlm_label_are_machine_labels_and_are_written(
    world: World, tmp_path: Path
) -> None:
    asyncio.run(drive(world, tmp_path / 'loop', FakeVlm(world.truth)))

    for name in WRITTEN:
        doc = world.doc(name)
        assert doc['class_name'] == 'car', name
        assert doc['class_source'] in ('assist_propagated', 'vlm'), name


def test_boxes_are_never_touched(world: World, tmp_path: Path) -> None:
    asyncio.run(drive(world, tmp_path / 'loop', FakeVlm(world.truth)))

    for doc in world.os.docs(ITEMS).values():
        assert doc['region_boxes'] == BOXES


def test_a_human_relabel_after_the_snapshot_wins(world: World, tmp_path: Path) -> None:
    vlm = FakeVlm(world.truth)
    target = 'free-car-0199'

    def human_relabels(_ids: list[str]) -> None:
        doc = world.doc(target)
        if target not in vlm.asked and doc['class_source'] == 'item_proposal':
            doc.update(class_name='bus', class_id=world.class_id('bus'), class_source='human')
            doc['class_validated'] = True

    vlm.before_answer = human_relabels

    asyncio.run(drive(world, tmp_path / 'loop', vlm))

    doc = world.doc(target)
    assert (doc['class_name'], doc['class_source']) == ('bus', 'human')


@pytest.mark.parametrize(
    'change',
    [
        {'class_excluded': True},
        {'class_name': 'bus', 'class_source': 'vlm', 'label_source': 'vlm'},
    ],
    ids=['excluded', 'machine_relabel'],
)
def test_an_unlocked_change_after_the_snapshot_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    """Nothing locks the item, but its class state is not the one the node decided on."""
    world = make_world(tmp_path, monkeypatch)
    eligible = world.add_cluster(ROOT, {'car': 200}, prefix='free')
    k1 = required_sample(len(eligible), 0.9, 0.025, 0)
    sampled = set(child_sample([], sorted(eligible), k1, seed=0, path='r10001').draw)
    target = next(c for c in sorted(eligible) if c not in sampled)
    vlm = FakeVlm(world.truth)

    def someone_changes_it(_ids: list[str]) -> None:
        world.doc(target).update(change)

    vlm.before_answer = someone_changes_it

    run = asyncio.run(drive(world, tmp_path / 'loop', vlm, max_depth=0))

    (propagated,) = [e for e in run.store.events() if e['event'] == 'propagated']
    assert target not in propagated['written_ids']
    assert propagated['skipped']['token_changed'] == 1
    assert world.doc(target)['class_source'] != 'assist_propagated'


def test_a_human_write_landing_between_read_and_write_wins(world: World) -> None:
    """The OCC write re-reads after a conflict and finds the item locked."""
    ids = [f'free-car-{i:04d}' for i in range(10)]
    tokens = {cid: class_state_token(world.doc(cid)) for cid in ids}
    racer = ids[3]
    real_bulk = world.os.bulk
    raced = False

    async def bulk_with_race(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        nonlocal raced
        if not raced:
            raced = True
            doc = world.doc(racer)
            doc.update(class_name='bus', class_id=world.class_id('bus'), class_source='human')
            world.os._bump(ITEMS, racer)
        return await real_bulk(body=body, **kw)

    world.os.bulk = bulk_with_race  # type: ignore[method-assign]
    label = ItemLabel.propagated(
        class_id=world.class_id('car'), class_name='car', run_id='alp1', writer='assist_loop:alp1'
    )

    result = asyncio.run(
        propagate_node(
            world.os,  # type: ignore[arg-type]
            ClusterFieldPartition(world.os),  # type: ignore[arg-type]
            root_key=root_path(ROOT),
            eligible=ids,
            answers={},
            tokens=tokens,
            label=label,
        )
    )

    assert racer not in result.written_ids
    assert result.skipped['locked'] == 1
    assert world.doc(racer)['class_source'] == 'human'
    assert all(world.doc(cid)['class_source'] == 'assist_propagated' for cid in ids if cid != racer)


def test_propagation_writes_the_machine_kind_with_the_run_in_its_provenance(
    world: World,
) -> None:
    ids = ['suggested_import', 'vlm_labeled', 'free-car-0000']
    tokens = {cid: class_state_token(world.doc(cid)) for cid in ids}
    label = ItemLabel.propagated(
        class_id=world.class_id('car'), class_name='car', run_id='alp1', writer='assist_loop:alp1'
    )

    result = asyncio.run(
        propagate_node(
            world.os,  # type: ignore[arg-type]
            ClusterFieldPartition(world.os),  # type: ignore[arg-type]
            root_key=root_path(ROOT),
            eligible=ids,
            answers={},
            tokens=tokens,
            label=label,
        )
    )

    assert sorted(result.written_ids) == sorted(ids)
    for cid in ids:
        doc = world.doc(cid)
        assert doc['class_source'] == 'assist_propagated'
        assert doc['class_validated'] is False
        assert doc['class_detector_version'] == 'alp1'
        assert doc['class_id_history'][-1]['writer'] == 'assist_loop:alp1'
        assert doc['region_boxes'] == BOXES
