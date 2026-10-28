"""An import whose image content another writer stored after the import's
lookup (an unrefreshed index, #358) attaches its labels to that stored image:
no second images doc, and no labels dropped with the refused create."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.dataset_import.harness import Harness, map_all, write_yolo
from src.services.curation.ingest_helpers import _imohash_bytes
from src.services.curation.ingest_index import content_image_id


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_import_lands_on_the_copy_stored_after_its_lookup(
    tmp_path: Path, monkeypatch
) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car'], images={'a': ['0 0.3 0.3 0.2 0.2']})
    data = (root / 'images/train/a.jpg').read_bytes()
    image_id = content_image_id(_imohash_bytes(data))
    stored = {
        'image_id': image_id,
        'image_path': '/elsewhere/a.jpg',
        'imohash': _imohash_bytes(data),
    }
    h.os.docs(h.cfg.images_index)[image_id] = dict(stored)

    async def unrefreshed_lookup(_image_hash: str) -> dict[str, Any] | None:
        return None

    monkeypatch.setattr(h.service, '_lookup_image', unrefreshed_lookup)

    store, _ = await h.run(h.request(root, map_all(h.registry, 'car')))

    assert store.job.read()['status'] == 'completed'
    assert set(h.images) == {image_id}
    assert h.images[image_id]['image_path'] == '/elsewhere/a.jpg'
    (item,) = h.items.values()
    assert (item['image_id'], item['image_path']) == (image_id, '/elsewhere/a.jpg')
    assert (item['class_name'], item['label_source']) == ('car', 'import')
    report = store.job.read()['report']
    assert (report['images_created'], report['images_reused']) == (0, 1)
    (row,) = store.ledger_rows()
    assert (row['status'], row['image_created'], row['image_id']) == ('ok', False, image_id)
