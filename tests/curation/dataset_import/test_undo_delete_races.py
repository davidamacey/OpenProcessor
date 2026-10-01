"""Undo deletes only what is still import-only at the moment of the delete:
a human edit that lands after the decision keeps the item."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.dataset_import.harness import Harness, map_all, write_yolo
from src.services.curation.dataset_import import undo as undo_mod
from src.services.curation.dataset_import.undo import undo_import


if TYPE_CHECKING:
    from pathlib import Path

HUMAN = {'class_source': 'human', 'label_source': 'human', 'class_validated': True}


async def _imported(tmp_path: Path, monkeypatch: Any) -> tuple[Harness, Any]:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car'], images={'a': ['0 0.3 0.3 0.2 0.2'], 'b': ['0 0.5 0.5 0.3 0.3']})
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car')))
    return h, store


@pytest.mark.asyncio
async def test_undo_keeps_an_item_a_human_edited_after_the_decision(tmp_path, monkeypatch) -> None:
    h, store = await _imported(tmp_path, monkeypatch)
    real = undo_mod.delete_items
    victim = next(iter(h.items))

    async def human_edit_then_delete(os_client: Any, ids: list[str], **kw: Any) -> dict[str, Any]:
        h.items[victim].update(HUMAN)
        return await real(os_client, ids, **kw)

    monkeypatch.setattr(undo_mod, 'delete_items', human_edit_then_delete)
    report = await undo_import(h.undo_context(store.import_id), store, dry_run=False)
    assert victim in h.items
    assert h.items[victim]['import_ids'] == []
    assert report.items_kept_human_edited == 1
    assert report.items_deleted == 1


@pytest.mark.asyncio
async def test_undo_keeps_an_item_edited_between_the_reread_and_the_delete(
    tmp_path, monkeypatch
) -> None:
    h, store = await _imported(tmp_path, monkeypatch)
    victim = next(iter(h.items))
    idx = h.cfg.items_index
    real_bulk = h.os.bulk

    async def edit_then_bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        # a write after the fresh read: same body, version bumped
        if any('delete' in a for a in body) and victim in h.items:
            h.items[victim].update(HUMAN)
            h.os._bump(idx, victim)
        return await real_bulk(body=body, **kw)

    monkeypatch.setattr(h.os, 'bulk', edit_then_bulk)
    report = await undo_import(h.undo_context(store.import_id), store, dry_run=False)
    assert victim in h.items
    assert h.items[victim]['class_source'] == 'human'
    assert (report.items_deleted, report.items_kept_human_edited) == (1, 1)


@pytest.mark.asyncio
async def test_undo_keeps_an_image_that_gained_an_item_after_the_decision(
    tmp_path, monkeypatch
) -> None:
    h, store = await _imported(tmp_path, monkeypatch)
    real = undo_mod.delete_items
    image_id = next(iter(h.images))

    async def ingest_into_image(os_client: Any, ids: list[str], **kw: Any) -> dict[str, Any]:
        if kw['items_index'] == h.cfg.images_index:
            h.items['late'] = {'crop_id': 'late', 'image_id': image_id, 'class_source': 'human'}
        return await real(os_client, ids, **kw)

    monkeypatch.setattr(undo_mod, 'delete_items', ingest_into_image)
    await undo_import(h.undo_context(store.import_id), store, dry_run=False)
    assert image_id in h.images
