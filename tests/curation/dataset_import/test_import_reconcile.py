"""A newer dataset version over an earlier import (W10.11): items it no
longer has go, unless a human or a holdout freeze touched them, and undoing
the newer import puts them back."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.dataset_import.harness import Harness, map_all, write_yolo
from src.config.region_fields import get_region_fields
from src.services.curation.dataset_import import reconcile
from src.services.curation.dataset_import.mapping import ClassMappingEntry
from src.services.curation.dataset_import.scan import ScanEntry
from src.services.curation.dataset_import.undo import undo_import


if TYPE_CHECKING:
    from pathlib import Path

CAR = '0 0.3 0.3 0.2 0.2'
TRUCK = '1 0.7 0.7 0.2 0.2'


async def _v1(tmp_path: Path, monkeypatch: Any) -> tuple[Harness, Path]:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    h.registry.add_class('truck')
    v1 = tmp_path / 'v1'
    write_yolo(v1, names=['car', 'truck'], images={'a': [CAR, TRUCK]})
    await h.run(h.request(v1, map_all(h.registry, 'car', 'truck')))
    assert len(h.items) == 2
    return h, tmp_path


def _truck(h: Harness) -> str:
    return next(cid for cid, d in h.items.items() if d['class_name'] == 'truck')


def _v2(root: Path, rows: list[str] | None, name: str = 'v2') -> Path:
    path = root / name
    write_yolo(path, names=['car', 'truck'], images={'a': rows})
    return path


def _names(h: Harness) -> list[str]:
    return sorted(d['class_name'] for d in h.items.values())


@pytest.mark.asyncio
async def test_an_item_the_new_version_no_longer_has_is_removed(tmp_path, monkeypatch) -> None:
    h, root = await _v1(tmp_path, monkeypatch)
    store, _ = await h.run(h.request(_v2(root, [CAR]), map_all(h.registry, 'car')))
    assert _names(h) == ['car']
    assert store.job.read()['report']['items_reconciled_removed'] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['class_source', 'label_source'])
async def test_a_human_edited_item_is_kept(tmp_path, monkeypatch, field: str) -> None:
    h, root = await _v1(tmp_path, monkeypatch)
    h.items[_truck(h)][field] = 'human'
    await h.run(h.request(_v2(root, [CAR]), map_all(h.registry, 'car')))
    assert _names(h) == ['car', 'truck']


@pytest.mark.asyncio
async def test_an_item_with_a_human_region_box_is_kept(tmp_path, monkeypatch) -> None:
    h, root = await _v1(tmp_path, monkeypatch)
    human_box = {'box_id': 'b1', 'bbox_norm': [0.7, 0.7, 0.8, 0.8], 'state': 'accepted'}
    h.items[_truck(h)][get_region_fields().boxes] = [{**human_box, 'source': 'human'}]
    await h.run(h.request(_v2(root, [CAR]), map_all(h.registry, 'car')))
    assert _names(h) == ['car', 'truck']


@pytest.mark.asyncio
async def test_a_holdout_item_is_kept(tmp_path, monkeypatch) -> None:
    h, root = await _v1(tmp_path, monkeypatch)
    h.items[_truck(h)]['test_holdout'] = True
    await h.run(h.request(_v2(root, [CAR]), map_all(h.registry, 'car')))
    assert _names(h) == ['car', 'truck']


@pytest.mark.asyncio
async def test_a_box_mapped_to_skip_still_keeps_its_item(tmp_path, monkeypatch) -> None:
    h, root = await _v1(tmp_path, monkeypatch)
    mapping = [
        *map_all(h.registry, 'car'),
        ClassMappingEntry(dataset_class='truck', action='skip'),
    ]
    await h.run(h.request(_v2(root, [CAR, TRUCK]), mapping))
    assert _names(h) == ['car', 'truck']


@pytest.mark.asyncio
async def test_a_frame_without_a_label_file_removes_nothing(tmp_path, monkeypatch) -> None:
    h, root = await _v1(tmp_path, monkeypatch)
    await h.run(h.request(_v2(root, None), []))
    assert _names(h) == ['car', 'truck']


@pytest.mark.asyncio
async def test_a_human_edit_between_the_decision_and_the_delete_wins(tmp_path, monkeypatch) -> None:
    h, root = await _v1(tmp_path, monkeypatch)
    victim = _truck(h)
    real = reconcile.delete_items

    async def edit_then_delete(opensearch: Any, ids: list[str], **kw: Any) -> dict[str, Any]:
        h.items[victim].update({'class_source': 'human', 'label_source': 'human'})
        return await real(opensearch, ids, **kw)

    monkeypatch.setattr(reconcile, 'delete_items', edit_then_delete)
    await h.run(h.request(_v2(root, [CAR]), map_all(h.registry, 'car')))
    assert victim in h.items


@pytest.mark.asyncio
async def test_undoing_the_newer_import_puts_the_removed_item_back(tmp_path, monkeypatch) -> None:
    h, root = await _v1(tmp_path, monkeypatch)
    victim = _truck(h)
    original = dict(h.items[victim])
    store, _ = await h.run(h.request(_v2(root, [CAR]), map_all(h.registry, 'car')))
    assert victim not in h.items

    dry = await undo_import(h.undo_context(store.import_id), store, dry_run=True)
    assert dry.items_reinstated == 1
    assert victim not in h.items
    applied = await undo_import(h.undo_context(store.import_id), store, dry_run=False)
    assert applied.items_reinstated == 1
    # The fake index holds numpy floats, which the JSON ledger writes as text.
    vector = 'pe_embedding'
    back = h.items[victim]
    assert [float(x) for x in back.pop(vector)] == [float(x) for x in original.pop(vector)]
    assert back == original
    again = await undo_import(h.undo_context(store.import_id), store, dry_run=True)
    assert again.items_reinstated == 0


@pytest.mark.asyncio
async def test_a_redone_image_keeps_what_the_crashed_attempt_recorded(
    tmp_path, monkeypatch
) -> None:
    """The docs a crashed attempt already removed are gone from the index, so
    only the ledger row still names them."""
    h, root = await _v1(tmp_path, monkeypatch)
    _store, ctx = await h.run(h.request(_v2(root, [CAR]), map_all(h.registry, 'car')))
    recorded = {'crop_id': 'gone', 'action': reconcile.ACTION, 'doc': {'crop_id': 'gone'}}
    entry = ScanEntry(
        rel_path='a.jpg',
        source_stem='a',
        abs_image_path=root / 'a.jpg',
        split='train',
        label_state='labeled',
    )
    carried = await reconcile.dropped_entries(
        ctx, [], entry, [], image_id='img', prior={'items': [recorded]}
    )
    assert carried == [recorded]
