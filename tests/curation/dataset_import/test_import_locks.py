"""The lock rule inside an import: a human-owned label or box is never
written, whether the human edit landed before the plan or between the plan
and the write."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.dataset_import.harness import Harness, activate_region_profile, map_all, write_yolo
from src.config.region_fields import get_region_fields
from src.services.curation.dataset_import import runner
from src.services.curation.dataset_import.item_labels import apply_item_updates, plan_item_labels
from src.services.curation.dataset_import.mapping import ClassMappingEntry, MapTarget
from src.services.curation.dataset_import.scan import LabelBox
from src.services.curation.dataset_import.store import imports_root, open_store, valid_import_id
from src.services.curation.region_boxes import read_boxes


if TYPE_CHECKING:
    from pathlib import Path

BOX = (0.1, 0.1, 0.5, 0.5)
CAR = MapTarget(kind='item', class_id=0, class_name='car')


def _doc(**fields: Any) -> dict[str, Any]:
    from src.services.detection.geometry import crop_id

    return {
        'crop_id': crop_id('img', list(BOX)),
        'image_id': 'img',
        'bbox_norm': list(BOX),
        'class_id': 7,
        'class_name': 'bus',
        **fields,
    }


@pytest.mark.parametrize(
    'fields',
    [
        {'class_source': 'human', 'label_source': 'human'},
        {'class_source': 'human_move', 'label_source': 'human'},
        {'class_source': 'vlm', 'label_source': 'human'},
        {'class_source': 'vlm', 'label_source': 'vlm', 'test_holdout': True},
    ],
)
def test_a_human_owned_match_is_a_conflict_not_an_update(fields: dict[str, Any]) -> None:
    (plan,) = plan_item_labels([_doc(**fields)], [(LabelBox('car', BOX), CAR)], image_id='img')
    assert plan.action == 'locked_conflict'


def test_the_same_class_on_a_human_item_is_a_noop() -> None:
    doc = _doc(class_id=0, class_name='car', class_source='human', label_source='human')
    (plan,) = plan_item_labels([doc], [(LabelBox('car', BOX), CAR)], image_id='img')
    assert plan.action == 'noop'


def test_machine_and_older_import_items_are_updated_with_a_snapshot_slot() -> None:
    machine = _doc(class_source='vlm', label_source='vlm', class_id_history=[{'writer': 'x'}])
    (plan,) = plan_item_labels([machine], [(LabelBox('car', BOX), CAR)], image_id='img')
    assert (plan.action, plan.snapshot_index) == ('updated', 1)
    older_import = _doc(class_source='external_label', label_source='import', class_validated=True)
    (plan,) = plan_item_labels([older_import], [(LabelBox('car', BOX), CAR)], image_id='img')
    assert plan.action == 'updated'  # a newer dataset version may correct an older one


@pytest.mark.asyncio
async def test_a_human_edit_between_plan_and_write_still_wins(tmp_path: Path, monkeypatch) -> None:
    """The plan said 'updated'; by the time the OCC write reads the doc a
    human has relabeled it. The write must re-check, not trust the plan."""
    from src.services.curation.dataset_import.item_labels import ItemPlan

    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car'], images={'a': ['0 0.3 0.3 0.4 0.4']})
    prepared = h.prepare(h.request(root, map_all(h.registry, 'car')))
    request = h.request(root, map_all(h.registry, 'car'))
    store, _ = runner.claim_import(request, prepared)
    ctx = h.context(store, request, prepared)
    machine = _doc(class_source='vlm', label_source='vlm', class_validated=False)
    h.items[machine['crop_id']] = dict(machine)
    plan = ItemPlan(LabelBox('car', BOX), CAR, machine['crop_id'], 'updated', machine, 0, False)
    # The human edit lands after the plan was made.
    h.items[machine['crop_id']].update({'class_source': 'human', 'label_source': 'human'})
    entry = prepared.scan.entries[0]
    outcomes = await apply_item_updates(ctx, entry, [plan], '2026-01-01T00:00:00+00:00')
    assert outcomes == {machine['crop_id']: 'locked_conflict'}
    after = h.items[machine['crop_id']]
    assert (after['class_name'], after['class_source']) == ('bus', 'human')
    assert 'class_id_history' not in after


@pytest.mark.asyncio
async def test_region_boxes_are_never_written_over_a_human_box_or_verdict(
    tmp_path: Path, monkeypatch
) -> None:
    activate_region_profile(parent_classes=('car',))
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    mapping = [
        *map_all(h.registry, 'car'),
        ClassMappingEntry(dataset_class='wheel', action='region'),
    ]
    root = tmp_path / 'ds'
    write_yolo(root, names=['car', 'wheel'], images={'a': ['0 0.3 0.3 0.4 0.4']})
    await h.run(h.request(root, map_all(h.registry, 'car')))  # the parent exists, no boxes yet
    (cid,) = h.items
    F = get_region_fields()
    human_box = {
        'box_id': 'b1',
        'bbox_norm': [0.2, 0.2, 0.25, 0.25],
        'state': 'accepted',
        'source': 'human',
        'detector': 'human',
    }
    h.items[cid][F.boxes] = [human_box]
    root2 = tmp_path / 'ds2'
    write_yolo(
        root2, names=['car', 'wheel'], images={'a': ['0 0.3 0.3 0.4 0.4', '1 0.3 0.3 0.05 0.05']}
    )
    (root2 / 'images/train/a.jpg').write_bytes((root / 'images/train/a.jpg').read_bytes())
    store, _ = await h.run(h.request(root2, mapping))
    assert [b.source for b in read_boxes(h.items[cid], F)] == ['human']
    assert store.job.read()['report']['label_conflicts_locked'] == 1
    assert store.job.read()['report']['boxes_written'] == 0


def test_import_ids_are_only_ever_ones_the_store_generated(tmp_path: Path, monkeypatch) -> None:
    Harness(tmp_path, monkeypatch)
    root = imports_root()
    (root / 'imp_20260101T000000_aaaaaaaa').mkdir(parents=True)
    (root.parent / 'evil').mkdir(parents=True)
    assert open_store('imp_20260101T000000_aaaaaaaa') is not None
    for bad in ('../evil', '..', '.', '', 'evil', 'imp_20260101T000000_AAAAAAAA', '../../x'):
        assert valid_import_id(bad) is False
        assert open_store(bad) is None
