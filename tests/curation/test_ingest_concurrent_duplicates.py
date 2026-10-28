"""Identical content ingested concurrently is stored once (#358).

The imohash dedup lookup runs before the write, and a search does not see a
write until the index refreshes, so two requests carrying the same bytes can
both miss it. These tests hold every write until each copy's lookup has run,
which is the race the live campaign hit, and check that the write itself
refuses the second copy: one images doc, its items only, its crops only, and
the losing copy reported ``duplicate`` with the stored image's id.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import TYPE_CHECKING, Any

import pytest

from curation.ingest_fakes import _jpeg_bytes, _make_service
from curation.occ_fakes import FakeIngestOpenSearch
from src.config import get_curation_config
from src.services.curation.ingest_helpers import _imohash_bytes
from src.services.curation.ingest_index import content_image_id
from src.services.curation.ingest_writer import IngestWriter


if TYPE_CHECKING:
    from pathlib import Path

    from src.services.curation.ingest import CurationIngestService


def _hold_writes_until_lookups(os_fake: FakeIngestOpenSearch, n_lookups: int) -> None:
    """Every bulk waits until ``n_lookups`` imohash lookups have run: no copy can
    see another copy's write, as when the writes are not yet refreshed."""
    looked_up = asyncio.Event()
    count = 0
    real_search, real_bulk = os_fake.search, os_fake.bulk

    async def search(**kw: Any) -> Any:
        nonlocal count
        if 'imohash' in ((kw['body'].get('query') or {}).get('term') or {}):
            count += 1
            if count >= n_lookups:
                looked_up.set()
        return await real_search(**kw)

    async def bulk(**kw: Any) -> Any:
        await asyncio.wait_for(looked_up.wait(), timeout=10)
        return await real_bulk(**kw)

    os_fake.search = search  # type: ignore[method-assign]
    os_fake.bulk = bulk  # type: ignore[method-assign]


def _cache_in(svc: CurationIngestService, tmp_path: Path) -> Path:
    cache = tmp_path / 'crops'
    svc.config = dataclasses.replace(svc.config, crop_cache_dir=cache)
    return cache


def _assert_stored_once(os_fake: FakeIngestOpenSearch, cache: Path) -> None:
    hashes = [d['imohash'] for d in os_fake.images.values()]
    assert len(hashes) == len(set(hashes)), 'the same content is stored as two images'
    assert {d['image_id'] for d in os_fake.items.values()} <= set(os_fake.images)
    assert {p.stem for p in (cache / 'default').glob('*.jpg')} == set(os_fake.items)


@pytest.mark.asyncio
async def test_same_content_in_two_concurrent_batches_is_stored_once(tmp_path: Path) -> None:
    svc, os_fake, _ = _make_service()
    cache = _cache_in(svc, tmp_path)
    shared, only_a, only_b = (_jpeg_bytes(seed=s) for s in (1, 2, 3))
    _hold_writes_until_lookups(os_fake, n_lookups=4)

    batch_a, batch_b = await asyncio.gather(
        svc.ingest_batch([shared, only_a], ['/a/copy.jpg', '/a/two.jpg']),
        svc.ingest_batch([shared, only_b], ['/b/copy.jpg', '/b/three.jpg']),
    )

    assert len(os_fake.images) == 3
    _assert_stored_once(os_fake, cache)
    assert len(os_fake.items) == 3
    results = [*batch_a.results, *batch_b.results]
    assert batch_a.summary.successful + batch_b.summary.successful == 3
    assert batch_a.summary.duplicates + batch_b.summary.duplicates == 1
    assert batch_a.summary.crops_indexed + batch_b.summary.crops_indexed == 3
    assert {batch_a.status, batch_b.status} == {'success'}
    (dup,) = [r for r in results if r.status == 'duplicate']
    (kept,) = [r for r in results if r.status == 'success' and r.imohash == dup.imohash]
    assert dup.image_id == kept.image_id
    assert dup.image_id in os_fake.images
    assert (dup.n_crops, dup.crops_created) == (0, 0)


@pytest.mark.asyncio
async def test_a_losing_copy_with_other_boxes_leaves_no_items_or_crops(tmp_path: Path) -> None:
    """Inference on the same bytes need not give the same boxes (another batch
    shape, another kernel): the losing copy's own items and crops must not stay."""
    svc, os_fake, triton = _make_service()
    cache = _cache_in(svc, tmp_path)
    boxes = iter([(0.05, 0.05, 0.6, 0.6, 0.9, 1), (0.1, 0.1, 0.7, 0.7, 0.9, 1)])
    real_infer = triton.infer

    async def infer(model_name: str, inputs: list, outputs: list) -> Any:
        triton.detections = [next(boxes)]
        return await real_infer(model_name, inputs, outputs)

    triton.infer = infer  # type: ignore[method-assign]
    _hold_writes_until_lookups(os_fake, n_lookups=2)
    data = _jpeg_bytes(seed=7)

    first, second = await asyncio.gather(
        svc.ingest_one(data, '/a/copy.jpg'), svc.ingest_one(data, '/b/copy.jpg')
    )

    assert sorted([first.status, second.status]) == ['duplicate', 'success']
    winner = first if first.status == 'success' else second
    assert len(os_fake.images) == 1
    assert len(os_fake.items) == 1
    assert winner.crops_created == 1
    _assert_stored_once(os_fake, cache)


@pytest.mark.asyncio
async def test_an_in_batch_copy_follows_a_representative_that_lost() -> None:
    """Another process stored the content and the index has not refreshed: the
    batch's representative loses the create, and its in-batch copy is answered
    from it instead of being decoded and detected again."""
    svc, os_fake, triton = _make_service()
    data = _jpeg_bytes(seed=11)
    image_id = content_image_id(_imohash_bytes(data))
    os_fake.images[image_id] = {'image_id': image_id, 'image_path': '/elsewhere/x.jpg'}

    async def unrefreshed_search(**_kw: Any) -> Any:
        return {'hits': {'hits': []}}

    async def unrefreshed_msearch(*, body: list[Any]) -> Any:
        return {'responses': [{'hits': {'hits': []}}] * (len(body) // 2)}

    os_fake.search = unrefreshed_search  # type: ignore[method-assign]
    os_fake.msearch = unrefreshed_msearch  # type: ignore[method-assign]

    batch = await svc.ingest_batch([data, data], ['/b/x.jpg', '/b/x-copy.jpg'])

    assert set(os_fake.images) == {image_id}
    assert os_fake.items == {}
    assert (batch.summary.successful, batch.summary.duplicates) == (0, 2)
    assert [r.image_id for r in batch.results] == [image_id, image_id]
    assert triton.batch_sizes == [1]


def _unit_args(image_id: str | None, crop_ids: list[str]) -> tuple[Any, ...]:
    image_doc = {'image_id': image_id} if image_id else None
    return image_doc, [{'crop_id': c, 'image_id': image_id or 'x'} for c in crop_ids]


@pytest.mark.asyncio
async def test_the_writer_refuses_a_second_images_doc_and_drops_its_items() -> None:
    os_fake = FakeIngestOpenSearch()
    writer = IngestWriter(os_fake)
    cfg = get_curation_config()
    kw = {'images_index': cfg.images_index, 'items_index': cfg.items_index, 'guards': ()}

    first, second = await asyncio.gather(
        writer.write(*_unit_args('img', ['a1']), **kw),
        writer.write(*_unit_args('img', ['b1', 'b2']), **kw),
    )

    assert set(os_fake.images) == {'img'}
    assert set(os_fake.items) == {'a1'}
    assert writer.bulk_sizes == [5]
    assert (first.images_indexed, first.created, first.duplicate) == (1, 1, False)
    assert (second.images_indexed, second.created, second.duplicate) == (0, 0, True)
    assert second.refused == []
