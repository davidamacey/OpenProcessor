"""What a repair may touch (plan 13a, lane L0, after the independent review).

A repair completes a photo a failed or killed ingest left without some of its items. It must not
become a second writer of the items that exist: it CREATES the missing ones (a 409 leaves the
stored item untouched, whoever wrote it), it skips every detection that overlaps an item the image
already has (the detector is not bit-stable and a crop id hashes the box to 6 decimals), and only
one repair runs on an image at a time (a lease on the images doc).
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from opensearchpy.exceptions import TransportError

from curation.ingest_fakes import _jpeg_bytes, _make_service
from curation.occ_fakes import FakeIngestOpenSearch
from curation.test_ingest_concurrent_duplicates import _hold_writes_until_lookups
from curation.test_ingest_whole_photo import PATH, TWO_BOXES, _image_id, _old, _writes_done
from src.clients.image_counts_dirty import CountsMarkError
from src.config import get_curation_config
from src.services.curation import ingest_writer
from src.services.curation.ingest_writer import REPAIR_AFTER_S, IngestWriter, RepairTarget
from src.services.curation.item_delete import delete_items
from src.services.near_duplicates.content_key import content_imohash


def _shifted(boxes: list[tuple[float, ...]], by: float) -> list[tuple[float, ...]]:
    """The same objects as the detector sees them on another run: boxes a hair apart, so the
    6-decimal crop id differs."""
    return [(x1 + by, y1 + by, x2 + by, y2 + by, s, c) for x1, y1, x2, y2, s, c in boxes]


async def _incomplete(
    svc: Any, os_fake: FakeIngestOpenSearch, data: bytes, *, keep: int = 1
) -> tuple[str, list[str], list[str]]:
    """Ingest ``data`` whole, then lose all but the first ``keep`` items (a kill between the two
    bulks, or a failed items step) and make the doc older than a live writer can be.
    Returns ``(image id, surviving ids, lost ids)``."""
    await svc.ingest_one(data, PATH)
    image_id = _image_id(data)
    ids = sorted(os_fake.items)
    lost = ids[keep:]
    for item_id in lost:
        del os_fake.items[item_id]
    _old(os_fake, image_id)
    return image_id, ids[:keep], lost


@pytest.mark.asyncio
async def test_a_repair_never_rewrites_an_item_that_exists_holdout_without_a_class() -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=61)
    _, (survivor,), _ = await _incomplete(svc, os_fake, data)
    doc = os_fake.items[survivor]
    for field in ('class_source', 'label_source', 'class_validated', 'class_id', 'class_name'):
        doc.pop(field, None)
    doc.update(test_holdout=True, bbox_norm=[0.06, 0.06, 0.41, 0.41], blur_lap_var=123.0)
    before = copy.deepcopy(doc)

    assert (await svc.ingest_one(data, PATH)).status == 'success'

    assert os_fake.items[survivor] == before


@pytest.mark.asyncio
async def test_a_repair_keeps_a_box_a_person_moved_and_its_vector() -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=62)
    _, (survivor,), _ = await _incomplete(svc, os_fake, data)
    os_fake.items[survivor].update(
        geometry_source='human', bbox_norm=[0.07, 0.07, 0.42, 0.42], pe_embedding=[9.0, 9.0, 9.0]
    )
    before = copy.deepcopy(os_fake.items[survivor])

    assert (await svc.ingest_one(data, PATH)).status == 'success'

    assert os_fake.items[survivor] == before


@pytest.mark.asyncio
async def test_a_repair_keeps_the_machine_work_of_an_item_that_exists() -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=63)
    _, (survivor,), lost = await _incomplete(svc, os_fake, data)
    os_fake.items[survivor].update(vlm_class_name='machine-label', cluster_uid='c-1')
    before = copy.deepcopy(os_fake.items[survivor])

    assert (await svc.ingest_one(data, PATH)).status == 'success'

    assert os_fake.items[survivor] == before
    assert set(lost) <= set(os_fake.items), 'the lost items are created again'


@pytest.mark.asyncio
async def test_a_detection_that_differs_in_the_sixth_decimal_creates_no_second_item() -> None:
    svc, os_fake, triton = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=64)
    image_id, (survivor,), lost = await _incomplete(svc, os_fake, data)
    triton.detections = _shifted(TWO_BOXES, 3e-5)  # the same objects, other crop ids

    assert (await svc.ingest_one(data, PATH)).status == 'success'

    assert survivor in os_fake.items, 'the stored item is kept, not replaced by a new id'
    assert len(os_fake.items) == 2, 'one item per object'
    assert lost[0] not in os_fake.items, 'the lost object is created once, under its new id'
    assert sorted(os_fake.items) == os_fake.images[image_id]['item_ids']


@pytest.mark.asyncio
async def test_two_repairs_of_one_image_at_once_leave_one_item_per_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc, os_fake, triton = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=65)
    image_id, _, _ = await _incomplete(svc, os_fake, data, keep=0)
    runs = iter(range(1, 100))
    real_infer = triton.infer

    async def infer(model_name: str, inputs: list, outputs: list) -> Any:
        triton.detections = _shifted(TWO_BOXES, next(runs) * 2e-5)  # each run: other ids
        return await real_infer(model_name, inputs, outputs)

    triton.infer = infer
    _hold_writes_until_lookups(os_fake, 2, monkeypatch)

    first, second = await asyncio.gather(
        svc.ingest_one(data, '/a/x.jpg'), svc.ingest_one(data, '/b/x.jpg')
    )

    assert sorted((first.status, second.status)) == ['failed', 'success']
    (held,) = [r for r in (first, second) if r.status == 'failed']
    assert held.error_kind == 'incomplete_write'
    assert len(os_fake.items) == 2, 'one item per object, not one set per repairer'
    assert sorted(os_fake.items) == os_fake.images[image_id]['item_ids']
    assert not os_fake.images[image_id].get('repair_started_at')


@pytest.mark.asyncio
async def test_a_repair_in_progress_holds_the_image_even_when_it_is_marked() -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=66)
    image_id, _, _ = await _incomplete(svc, os_fake, data, keep=0)
    os_fake.images[image_id].update(
        ingest_incomplete=True, repair_started_at=datetime.now(UTC).isoformat()
    )

    held = await svc.ingest_one(data, PATH)

    assert (held.status, held.error_kind) == ('failed', 'incomplete_write')
    assert os_fake.items == {}
    os_fake.images[image_id]['repair_started_at'] = (
        datetime.now(UTC) - timedelta(seconds=REPAIR_AFTER_S + 60)
    ).isoformat()
    assert (await svc.ingest_one(data, PATH)).status == 'success', 'an old lease is a dead repair'


@pytest.mark.asyncio
async def test_a_repair_that_fails_gives_the_lease_back_and_marks_the_photo() -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=67)
    image_id, _, _ = await _incomplete(svc, os_fake, data, keep=0)
    real = os_fake.bulk
    left = {'n': 1}

    async def bulk(*, body: list[dict[str, Any]], **kw: Any) -> Any:
        items_index = get_curation_config().items_index
        if left['n'] and any(a.get('create', {}).get('_index') == items_index for a in body[0::2]):
            left['n'] -= 1
            raise TransportError(500, 'repair items bulk failed')
        return await real(body=body, **kw)

    os_fake.bulk = bulk

    failed = await svc.ingest_one(data, PATH)
    await _writes_done(svc)

    assert (failed.status, failed.error_kind) == ('failed', 'bulk_index')
    doc = os_fake.images[image_id]
    assert doc['ingest_incomplete'] is True
    assert not doc.get('repair_started_at')
    assert (await svc.ingest_one(data, PATH)).status == 'success', 'the retry need not wait'


@pytest.mark.asyncio
async def test_an_item_deleted_while_it_is_repaired_is_not_listed_again() -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=68)
    image_id, _, _ = await _incomplete(svc, os_fake, data, keep=0)
    real = os_fake.bulk
    deleted: list[str] = []

    async def bulk(*, body: list[dict[str, Any]], **kw: Any) -> Any:
        resp = await real(body=body, **kw)
        items_index = get_curation_config().items_index
        made = [
            a['create']['_id']
            for a in body[0::2]
            if a.get('create', {}).get('_index') == items_index
        ]
        if made and not deleted:  # a re-detection removes one of them at once
            deleted.append(made[0])
            os_fake.bulk = real
            await delete_items(
                os_fake, [made[0]], items_index=items_index, crop_cache_dir=None,
                deletable=_always,
            )  # fmt: skip
            os_fake.bulk = bulk
        return resp

    os_fake.bulk = bulk

    assert (await svc.ingest_one(data, PATH)).status == 'success'

    (gone,) = deleted
    assert gone not in os_fake.items
    assert gone not in os_fake.images[image_id]['item_ids']
    os_fake.bulk = real
    assert (await svc.ingest_one(data, PATH)).status == 'duplicate', 'not repaired again'


async def _always(_crop_id: str, _doc: dict[str, Any]) -> bool:
    return True


def _kw() -> dict[str, Any]:
    cfg = get_curation_config()
    return {'images_index': cfg.images_index, 'items_index': cfg.items_index, 'guards': ()}


def _doc(crop_id: str, image_id: str) -> dict[str, Any]:
    return {'crop_id': crop_id, 'image_id': image_id, 'bbox_norm': [0.1, 0.1, 0.4, 0.4]}


@pytest.mark.asyncio
async def test_one_units_failed_repair_does_not_fail_a_whole_photo_of_its_flush() -> None:
    old = (datetime.now(UTC) - timedelta(seconds=REPAIR_AFTER_S + 60)).isoformat()
    os_fake = FakeIngestOpenSearch(
        images={'b': {'image_id': 'b', 'item_ids': ['b0'], 'indexed_at': old}}
    )
    real = os_fake.bulk

    async def bulk(*, body: list[dict[str, Any]], **kw: Any) -> Any:
        if body[0].get('update', {}).get('_id') == 'b':
            raise TransportError(503, 'the repair of b cannot be written')
        return await real(body=body, **kw)

    os_fake.bulk = bulk
    writer = IngestWriter(os_fake)

    whole, broken = await asyncio.gather(
        writer.write({'image_id': 'a'}, [_doc('a1', 'a')], **_kw()),
        writer.write(None, [_doc('b1', 'b')], repair=RepairTarget('b'), **_kw()),
    )

    assert whole.failure is None
    assert whole.refused == []
    assert broken.failure is not None
    assert 'a1' in os_fake.items


@pytest.mark.asyncio
async def test_a_hanging_mark_does_not_hold_back_the_answers_of_its_flush() -> None:
    os_fake = FakeIngestOpenSearch()
    real = os_fake.bulk
    release = asyncio.Event()
    items_index = get_curation_config().items_index

    async def bulk(*, body: list[dict[str, Any]], **kw: Any) -> Any:
        if any(a.get('create', {}).get('_index') == items_index for a in body[0::2]):
            raise TransportError(500, 'items bulk failed')
        if body[0].get('update') and body[1]['doc'].get('ingest_incomplete') is True:
            await release.wait()  # the store hangs on the mark
        return await real(body=body, **kw)

    os_fake.bulk = bulk
    writer = IngestWriter(os_fake)

    try:
        result = await asyncio.wait_for(
            writer.write({'image_id': 'a'}, [_doc('a1', 'a')], **_kw()), timeout=3
        )
    finally:
        release.set()
        await writer.close()

    assert result.failure is not None


@pytest.mark.asyncio
async def test_a_holder_found_at_claim_time_is_a_duplicate_only_when_it_is_whole(
    content_rows: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another copy claimed between this photo's lookup and its claim; its images doc misses
    items: the answer is not ``duplicate``."""
    from src.services.curation.ingest_dedup import Resolved

    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=69)
    image_id = _image_id(data)
    os_fake.images[image_id] = {
        'image_id': image_id,
        'image_path': '/elsewhere.jpg',
        'item_ids': ['their-item'],
        'indexed_at': datetime.now(UTC).isoformat(),
    }
    content_rows.rows[content_imohash(data)] = image_id

    async def lookup_that_missed(imohash: str, _data: bytes) -> Resolved:
        return Resolved(None, imohash)

    monkeypatch.setattr(svc, '_lookup_image', lookup_that_missed)

    result = await svc.ingest_one(data, PATH)

    assert (result.status, result.error_kind) == ('failed', 'incomplete_write')
    os_fake.items['their-item'] = {'crop_id': 'their-item', 'image_id': image_id}
    assert (await svc.ingest_one(data, PATH)).status == 'duplicate'


def _cache_in(svc: Any, tmp_path: Any) -> Any:
    cache = tmp_path / 'crops'
    svc.config = dataclasses.replace(svc.config, crop_cache_dir=cache)
    return cache / 'default'


@pytest.mark.asyncio
async def test_an_item_written_a_moment_ago_and_not_listed_is_matched_too() -> None:
    """A person drew a box over an object a second before the repair: the search that finds the
    image's unlisted items does not see a write the index has not refreshed yet, so the repair
    publishes the index first; otherwise the matching detection is created beside the drawn box."""
    svc, os_fake, _ = _make_service(
        detections=TWO_BOXES, opensearch=FakeIngestOpenSearch(near_real_time_search=True)
    )
    data = _jpeg_bytes(seed=70)
    await svc.ingest_one(data, PATH)
    image_id = _image_id(data)
    first, *_ = sorted(os_fake.items)
    drawn_box = os_fake.items[first]['bbox_norm']
    os_fake.items.clear()
    _old(os_fake, image_id)
    os_fake.items['drawn-1'] = {
        'crop_id': 'drawn-1',
        'image_id': image_id,
        'bbox_norm': drawn_box,
        'geometry_source': 'human',
    }

    assert (await svc.ingest_one(data, PATH)).status == 'success'

    assert 'drawn-1' in os_fake.items
    assert len(os_fake.items) == 2, 'the drawn box stands for its object; the other is created'


@pytest.mark.asyncio
async def test_a_counts_mark_error_after_the_items_exist_does_not_fail_the_photo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The items are stored; only the note that the image's counts need a recount could not be
    written (Valkey and the images index both refused). The photo is whole, so it answers
    ``success`` (the error is logged); failing it would turn it into a ``duplicate`` on the retry
    and lose its events and crop cache."""
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=71)
    image_id, _, _ = await _incomplete(svc, os_fake, data, keep=0)

    async def refuse(*_a: Any, **_k: Any) -> None:
        raise CountsMarkError('neither Valkey nor the images index took the mark')

    monkeypatch.setattr(ingest_writer, 'mark_items_created', refuse)

    assert (await svc.ingest_one(data, PATH)).status == 'success'
    await _writes_done(svc)

    assert len(os_fake.items) == 2
    assert not os_fake.images[image_id].get('repair_started_at'), 'the lease went back'


@pytest.mark.asyncio
async def test_an_unexpected_error_in_a_repair_gives_the_lease_back_and_marks_the_photo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=72)
    image_id, _, _ = await _incomplete(svc, os_fake, data, keep=0)
    real = IngestWriter._create_missing

    async def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError('a bug in the create step')

    monkeypatch.setattr(IngestWriter, '_create_missing', boom)

    failed = await svc.ingest_one(data, PATH)  # the error reaches the photo as a failed answer
    await _writes_done(svc)

    assert (failed.status, failed.error_kind) == ('failed', 'bulk_index')

    doc = os_fake.images[image_id]
    assert not doc.get('repair_started_at'), 'the lease is not held for the repair age'
    assert doc['ingest_incomplete'] is True
    monkeypatch.setattr(IngestWriter, '_create_missing', real)
    assert (await svc.ingest_one(data, PATH)).status == 'success', 'the retry need not wait'


@pytest.mark.asyncio
async def test_an_unexpected_error_before_the_lease_marks_the_photo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=73)
    image_id, _, _ = await _incomplete(svc, os_fake, data, keep=0)

    async def boom(*_a: Any, **_k: Any) -> None:
        raise KeyError('a bug in the match')

    monkeypatch.setattr(ingest_writer, 'plan_repair', boom)

    failed = await svc.ingest_one(data, PATH)
    await _writes_done(svc)

    assert (failed.status, failed.error_kind) == ('failed', 'bulk_index')

    assert os_fake.images[image_id]['ingest_incomplete'] is True
    assert not os_fake.images[image_id].get('repair_started_at')


@pytest.mark.asyncio
async def test_a_repair_that_matched_everything_reports_and_caches_nothing(tmp_path: Any) -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    cache = _cache_in(svc, tmp_path)
    data = _jpeg_bytes(seed=74)
    await svc.ingest_one(data, PATH)
    image_id = _image_id(data)
    os_fake.images[image_id]['item_ids'] = [*os_fake.images[image_id]['item_ids'], 'never-written']
    _old(os_fake, image_id)
    for path in cache.glob('*.jpg'):
        path.unlink()

    result = await svc.ingest_one(data, PATH)

    assert result.status == 'success'
    assert (result.n_crops, result.crops_created) == (0, 0)
    assert list(cache.glob('*.jpg')) == [], 'no crop file for an id that has no item'


@pytest.mark.asyncio
async def test_a_repair_reports_and_caches_the_items_it_created(tmp_path: Any) -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    cache = _cache_in(svc, tmp_path)
    data = _jpeg_bytes(seed=75)
    _, (survivor,), (lost,) = await _incomplete(svc, os_fake, data)
    for path in cache.glob('*.jpg'):
        path.unlink()

    result = await svc.ingest_one(data, PATH)

    assert (result.n_crops, result.crops_created) == (1, 1)
    assert {p.stem for p in cache.glob('*.jpg')} == {lost}
    assert survivor in os_fake.items


@pytest.mark.asyncio
async def test_a_second_repair_after_a_dead_one_leaves_one_item_per_object() -> None:
    """The first repairer died after its lease and list write and one create: its lease runs
    out, the second repair finds that item by overlap, creates the other, and the list names only
    items that exist."""
    svc, os_fake, triton = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=76)
    await svc.ingest_one(data, PATH)
    image_id = _image_id(data)
    first, second = sorted(os_fake.items)
    dead_first = os_fake.items.pop(first)
    dead_first.update(crop_id='dead-1')
    os_fake.items['dead-1'] = dead_first
    del os_fake.items[second]
    old = (datetime.now(UTC) - timedelta(seconds=REPAIR_AFTER_S + 60)).isoformat()
    os_fake.images[image_id].update(
        item_ids=['dead-1', 'dead-2'], repair_started_at=old, indexed_at=old
    )
    triton.detections = _shifted(TWO_BOXES, 4e-5)

    assert (await svc.ingest_one(data, PATH)).status == 'success'

    assert len(os_fake.items) == 2
    assert 'dead-1' in os_fake.items
    assert os_fake.images[image_id]['item_ids'] == sorted(os_fake.items)
    assert (await svc.ingest_one(data, PATH)).status == 'duplicate'


@pytest.mark.asyncio
async def test_two_overlapping_objects_are_both_kept_when_both_are_detected() -> None:
    """The match is one to one: an item takes its own detection (IoU 1), so the second object
    that overlaps it at IoU 0.5 or more is created, not mistaken for it."""
    outer, inner = (0.10, 0.10, 0.60, 0.60, 0.9, 1), (0.15, 0.15, 0.62, 0.62, 0.9, 1)
    svc, os_fake, _ = _make_service(detections=[outer, inner])
    data = _jpeg_bytes(seed=77)
    await svc.ingest_one(data, PATH)
    stored = sorted(os_fake.items.values(), key=lambda d: d['bbox_norm'])
    assert len(stored) == 2
    from src.services.detection.geometry import iou

    assert iou(*(tuple(d['bbox_norm']) for d in stored)) >= 0.5
    image_id = _image_id(data)
    del os_fake.items[stored[1]['crop_id']]
    _old(os_fake, image_id)

    assert (await svc.ingest_one(data, PATH)).status == 'success'

    assert sorted(os_fake.items) == sorted(d['crop_id'] for d in stored)


@pytest.mark.asyncio
async def test_a_holder_that_claimed_and_died_without_writing_is_stored_by_the_resend(
    content_rows: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation.ingest_dedup import Resolved

    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=78)
    image_id = _image_id(data)
    content_rows.rows[content_imohash(data)] = image_id  # claimed; no images doc, ever
    real_lookup = svc._lookup_image

    async def lookup_that_missed(imohash: str, _data: bytes) -> Resolved:
        return Resolved(None, imohash)

    monkeypatch.setattr(svc, '_lookup_image', lookup_that_missed)
    held = await svc.ingest_one(data, PATH)
    assert (held.status, held.error_kind) == ('failed', 'incomplete_write')
    assert os_fake.images == {}

    monkeypatch.setattr(svc, '_lookup_image', real_lookup)  # the resend looks the content up
    stored = await svc.ingest_one(data, PATH)

    assert stored.status == 'success', 'the lookup released the claim of a doc that never came'
    assert set(os_fake.images) == {image_id}
    assert len(os_fake.items) == 2


@pytest.mark.asyncio
async def test_a_repair_that_planned_before_another_finished_does_not_repair_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dangerous point of two repairs of one image: the second reads the image (nothing
    exists yet), the first then takes the lease, creates its items and gives the lease back, and
    only then the second reaches the lease. A lease that excludes only overlapping repairs lets
    the second one through with a plan made from a state that is gone, and it creates a second set
    of items under other crop ids (the detector is not bit-stable), one more item per object."""
    old = (datetime.now(UTC) - timedelta(seconds=REPAIR_AFTER_S + 60)).isoformat()
    os_fake = FakeIngestOpenSearch(
        images={'a': {'image_id': 'a', 'item_ids': ['a0', 'a1'], 'indexed_at': old}}
    )
    first_done = asyncio.Event()
    real_plan = ingest_writer.plan_repair

    async def plan(
        client: Any, items_index: str, image_id: str, listed: Any, crop_docs: Any
    ) -> Any:
        planned = await real_plan(client, items_index, image_id, listed, crop_docs)
        if crop_docs[0]['crop_id'].startswith('second'):
            await asyncio.wait_for(first_done.wait(), 10)
        return planned

    monkeypatch.setattr(ingest_writer, 'plan_repair', plan)

    def docs(prefix: str, by: float) -> list[dict[str, Any]]:
        return [
            {**_doc(f'{prefix}-{n}', 'a'), 'bbox_norm': [0.1 + n * 0.4 + by, 0.1, 0.4 + n * 0.4, 0.4]}
            for n in range(2)
        ]  # fmt: skip

    target = RepairTarget('a')
    first_writer, second_writer = IngestWriter(os_fake), IngestWriter(os_fake)
    first = asyncio.create_task(
        first_writer.write(None, docs('first', 0.0), repair=target, **_kw())
    )
    first.add_done_callback(lambda _t: first_done.set())
    second = asyncio.create_task(
        second_writer.write(None, docs('second', 2e-5), repair=target, **_kw())
    )

    results = [await first, await second]
    await first_writer.close()
    await second_writer.close()

    created = [r for r in results if r.created]
    assert len(created) == 1, 'exactly one repair creates items'
    (other,) = [r for r in results if r is not created[0]]
    assert other.incomplete or other.duplicate, 'the other one answers, it does not create'
    assert len(os_fake.items) == 2, 'one item per object, not one set per repairer'
    assert sorted(os_fake.items) == os_fake.images['a']['item_ids']
    assert not os_fake.images['a'].get('repair_started_at'), 'the lease went back'
