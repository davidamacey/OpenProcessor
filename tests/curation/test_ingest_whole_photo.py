"""A photo is stored whole or not at all (plan 13a, lane L0).

The images doc is written before its items, so a failed or killed items step leaves a doc with no
items. Before this lane the claim was released and the retry answered ``duplicate`` (or the lookup
did, after a kill): the photo sat in the library with no objects, and the bulk ledger marked the
file done. A refused item or images doc was only logged and the photo answered ``success``.

The images doc now lists its items (``item_ids``); a stored image is a ``duplicate`` only when all
of them exist, and an incomplete one is completed by the next ingest of the same bytes.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from opensearchpy.exceptions import TransportError

from curation.ingest_fakes import _jpeg_bytes, _make_service
from curation.occ_fakes import FakeIngestOpenSearch
from src.config import get_curation_config
from src.services.curation import ingest_writer, ops_metrics
from src.services.curation.ingest_writer import REPAIR_AFTER_S
from src.services.near_duplicates import content_key
from src.services.near_duplicates.content_key import content_image_id, content_imohash


TWO_BOXES = [(0.05, 0.05, 0.4, 0.4, 0.9, 1), (0.5, 0.5, 0.9, 0.9, 0.9, 1)]
PATH = '/library/photo.jpg'


def _image_id(data: bytes) -> str:
    return content_image_id(content_imohash(data))


def _is_items_write(body: list[dict[str, Any]]) -> bool:
    items_index = get_curation_config().items_index
    return any(a.get('create', {}).get('_index') == items_index for a in body[0::2])


def _fail_next_items_bulk(os_fake: FakeIngestOpenSearch, *, times: int = 1) -> None:
    """The next ``times`` items-index create bulks raise (a transport failure)."""
    real = os_fake.bulk
    left = {'n': times}

    async def bulk(*, body: list[dict[str, Any]], **kw: Any) -> Any:
        if left['n'] > 0 and _is_items_write(body):
            left['n'] -= 1
            raise TransportError(500, 'items bulk failed')
        return await real(body=body, **kw)

    os_fake.bulk = bulk


def _refuse_next_item(os_fake: FakeIngestOpenSearch) -> None:
    """The store refuses the first item document of the next items create bulk (HTTP 400)."""
    real = os_fake.bulk
    left = {'n': 1}

    async def bulk(*, body: list[dict[str, Any]], **kw: Any) -> Any:
        resp = await real(body=body, **kw)
        if left['n'] > 0 and _is_items_write(body):
            left['n'] -= 1
            first = resp['items'][0]['create']
            os_fake.items.pop(first['_id'])
            first.update(
                status=400, error={'type': 'mapper_parsing_exception', 'reason': 'refused'}
            )
            resp['errors'] = True
        return resp

    os_fake.bulk = bulk


def _old(os_fake: FakeIngestOpenSearch, image_id: str) -> None:
    """The images doc was written longer ago than a live writer can still be writing it."""
    when = datetime.now(UTC) - timedelta(seconds=REPAIR_AFTER_S + 60)
    os_fake.images[image_id]['indexed_at'] = when.isoformat()


@pytest.mark.asyncio
async def test_a_failed_items_step_keeps_the_claim_with_the_doc(content_rows: Any) -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=1)
    _fail_next_items_bulk(os_fake)

    first = await svc.ingest_one(data, PATH)

    assert (first.status, first.error_kind) == ('failed', 'bulk_index')
    image_id = _image_id(data)
    assert image_id in os_fake.images
    assert os_fake.items == {}
    assert content_rows.rows == {content_imohash(data): image_id}


@pytest.mark.asyncio
async def test_the_retry_of_a_failed_items_step_stores_the_photo_whole() -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=21)
    _fail_next_items_bulk(os_fake)
    await svc.ingest_one(data, PATH)

    second = await svc.ingest_one(data, PATH)

    assert second.status == 'success', 'the retry answered duplicate and stored no item'
    image_id = _image_id(data)
    assert len(os_fake.items) == 2
    assert len(os_fake.images) == 1
    assert sorted(os_fake.items) == sorted(os_fake.images[image_id]['item_ids'])
    assert (await svc.ingest_one(data, PATH)).status == 'duplicate'


@pytest.mark.asyncio
async def test_a_photo_whose_item_was_refused_answers_failed_and_a_retry_stores_it_whole() -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=2)
    _refuse_next_item(os_fake)

    first = await svc.ingest_one(data, PATH)

    assert (first.status, first.error_kind) == ('failed', 'bulk_index')
    assert len(os_fake.items) == 1, 'the item the store accepted stays'

    second = await svc.ingest_one(data, PATH)

    assert second.status == 'success'
    assert len(os_fake.items) == 2
    assert sorted(os_fake.items) == sorted(os_fake.images[_image_id(data)]['item_ids'])


@pytest.mark.asyncio
async def test_a_refused_images_doc_answers_failed_and_releases_the_claim(
    content_rows: Any,
) -> None:
    svc, os_fake, _ = _make_service()
    data = _jpeg_bytes(seed=3)
    real = os_fake.bulk

    async def refuse_images(*, body: list[dict[str, Any]], **kw: Any) -> Any:
        if not _is_items_write(body) and 'create' in body[0]:
            return {
                'errors': True,
                'items': [
                    {'create': {'_id': a['create']['_id'], 'status': 400, 'error': {'type': 'x'}}}
                    for a in body[0::2]
                ],
            }
        return await real(body=body, **kw)

    os_fake.bulk = refuse_images

    first = await svc.ingest_one(data, PATH)

    assert (first.status, first.error_kind) == ('failed', 'bulk_index')
    assert os_fake.images == {}
    assert content_rows.rows == {}, 'the doc is known absent: the content can be ingested again'
    os_fake.bulk = real
    assert (await svc.ingest_one(data, PATH)).status == 'success'


@pytest.mark.asyncio
async def test_a_photo_killed_between_the_two_bulks_is_held_back_then_repaired() -> None:
    """The writer died after the images doc: doc and claim, no items, no mark."""
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=4)
    await svc.ingest_one(data, PATH)
    image_id = _image_id(data)
    os_fake.items.clear()
    held_before = ops_metrics.OP_INGEST_WHOLE_TOTAL.labels(event='held_back')._value.get()

    young = await svc.ingest_one(data, PATH)

    assert (young.status, young.error_kind) == ('failed', 'incomplete_write')
    assert os_fake.items == {}
    assert ops_metrics.OP_INGEST_WHOLE_TOTAL.labels(event='held_back')._value.get() == (
        held_before + 1
    )

    _old(os_fake, image_id)
    repaired = await svc.ingest_one(data, PATH)

    assert repaired.status == 'success'
    assert len(os_fake.items) == 2
    assert len(os_fake.images) == 1
    assert (await svc.ingest_one(data, PATH)).status == 'duplicate'


@pytest.mark.asyncio
async def test_a_photo_the_writer_marked_is_repaired_at_once_and_the_mark_goes() -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=5)
    _fail_next_items_bulk(os_fake)
    await svc.ingest_one(data, PATH)
    image_id = _image_id(data)
    assert os_fake.images[image_id]['ingest_incomplete'] is True

    assert (await svc.ingest_one(data, PATH)).status == 'success'

    assert os_fake.images[image_id]['ingest_incomplete'] is False


@pytest.mark.asyncio
async def test_a_mark_that_cannot_be_written_changes_no_answer() -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=6)
    _fail_next_items_bulk(os_fake)
    real = os_fake.bulk

    async def no_marks(*, body: list[dict[str, Any]], **kw: Any) -> Any:
        if 'update' in body[0] and body[1]['doc'].get('ingest_incomplete') is True:
            raise TransportError(503, 'the mark write failed')
        return await real(body=body, **kw)

    os_fake.bulk = no_marks
    failed_before = ops_metrics.OP_INGEST_WHOLE_TOTAL.labels(event='mark_failed')._value.get()

    first = await svc.ingest_one(data, PATH)

    assert (first.status, first.error_kind) == ('failed', 'bulk_index')
    assert 'ingest_incomplete' not in os_fake.images[_image_id(data)]
    assert (
        ops_metrics.OP_INGEST_WHOLE_TOTAL.labels(event='mark_failed')._value.get() > failed_before
    )
    young = await svc.ingest_one(data, PATH)
    assert (young.status, young.error_kind) == ('failed', 'incomplete_write')
    _old(os_fake, _image_id(data))
    assert (await svc.ingest_one(data, PATH)).status == 'success'


@pytest.mark.asyncio
async def test_a_complete_stored_image_is_a_duplicate_with_one_more_read_and_no_write() -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=7)
    await svc.ingest_one(data, PATH)
    writes, reads = len(os_fake.bulk_calls), len(os_fake.mget_calls)

    again = await svc.ingest_one(data, '/elsewhere/copy.jpg')

    assert again.status == 'duplicate'
    assert len(os_fake.bulk_calls) == writes
    items_read = [ids for ids in os_fake.mget_calls[reads:] if ids and ids[0] in os_fake.items]
    assert len(items_read) == 1, 'one mget of the listed items'


@pytest.mark.asyncio
async def test_an_image_without_the_field_is_legacy_and_complete(content_rows: Any) -> None:
    """A doc written before the field (or copied by a combine) is a duplicate, never repaired."""
    svc, os_fake, _ = _make_service()
    data = _jpeg_bytes(seed=8)
    image_id = _image_id(data)
    os_fake.images[image_id] = {
        'image_id': image_id,
        'image_path': PATH,
        'indexed_at': (datetime.now(UTC) - timedelta(days=400)).isoformat(),
    }
    content_rows.rows[content_imohash(data)] = image_id

    result = await svc.ingest_one(data, '/other/copy.jpg')

    assert result.status == 'duplicate'
    assert os_fake.items == {}
    assert os_fake.bulk_calls == []


@pytest.mark.asyncio
async def test_an_image_listing_no_items_is_complete() -> None:
    svc, os_fake, _ = _make_service(detections=[])
    data = _jpeg_bytes(seed=9)
    assert (await svc.ingest_one(data, PATH)).status == 'success'
    assert os_fake.images[_image_id(data)]['item_ids'] == []

    assert (await svc.ingest_one(data, PATH)).status == 'duplicate'


@pytest.mark.asyncio
async def test_a_repair_deletes_a_stale_machine_item_and_keeps_a_human_one() -> None:
    svc, os_fake, triton = _make_service(detections=TWO_BOXES[:1])
    data = _jpeg_bytes(seed=10)
    await svc.ingest_one(data, PATH)
    image_id = _image_id(data)
    (stale_machine,) = os_fake.items
    os_fake.items['human-1'] = {
        'crop_id': 'human-1',
        'image_id': image_id,
        'class_source': 'human',
        'label_source': 'human',
    }
    os_fake.images[image_id]['item_ids'] = sorted([stale_machine, 'human-1', 'never-written'])
    _old(os_fake, image_id)
    triton.detections = TWO_BOXES[1:]  # another detector output: the old box is no longer found

    result = await svc.ingest_one(data, PATH)

    assert result.status == 'success'
    assert stale_machine not in os_fake.items
    assert 'human-1' in os_fake.items, 'a human-validated item is never deleted'
    (written,) = [i for i in os_fake.items if i not in ('human-1',)]
    assert os_fake.images[image_id]['item_ids'] == [written]


@pytest.mark.asyncio
async def test_a_repair_never_rewrites_the_images_doc_beyond_its_list() -> None:
    """A person or an import stamped the image after the first ingest: the repair keeps it."""
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=11)
    await svc.ingest_one(data, PATH)
    image_id = _image_id(data)
    os_fake.items.clear()
    os_fake.images[image_id].update(import_ids=['job-1'], dataset_split='val', negative_for=['x'])
    before = {k: v for k, v in os_fake.images[image_id].items() if k != 'item_ids'}
    _old(os_fake, image_id)
    before['indexed_at'] = os_fake.images[image_id]['indexed_at']

    assert (await svc.ingest_one(data, PATH)).status == 'success'

    after = {k: v for k, v in os_fake.images[image_id].items() if k != 'item_ids'}
    assert after == before


@pytest.mark.asyncio
async def test_a_re_detected_or_imported_image_is_not_repaired() -> None:
    """``adopt_existing`` (a dataset import, a reprocess) answers as before and writes no list."""
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=12)
    await svc.ingest_one(data, PATH)
    image_id = _image_id(data)
    os_fake.items.clear()
    _old(os_fake, image_id)
    listed = list(os_fake.images[image_id]['item_ids'])

    ctx = await svc.ingest_image(data, PATH, 'reprocess', adopt_existing=True)

    assert not isinstance(ctx, ingest_result_type())
    assert ctx.created is False
    assert ctx.repair is None
    outcome = await svc.index_items(ctx, [])
    assert outcome.result.status == 'success'
    assert os_fake.images[image_id]['item_ids'] == listed
    assert 'ingest_incomplete' not in os_fake.images[image_id]


def ingest_result_type() -> type:
    from src.services.curation.ingest_models import IngestResult

    return IngestResult


@pytest.mark.asyncio
async def test_a_409_on_an_incomplete_young_doc_answers_incomplete_and_keeps_the_claim(
    content_rows: Any,
) -> None:
    """Another process created the images doc and its items are not there yet."""
    svc, os_fake, _ = _make_service()
    data = _jpeg_bytes(seed=13)
    image_id = _image_id(data)
    os_fake.images[image_id] = {
        'image_id': image_id,
        'image_path': '/elsewhere/x.jpg',
        'ingest_write_id': 'another-writer',
        'indexed_at': datetime.now(UTC).isoformat(),
        'item_ids': ['their-item'],
    }

    result = await svc.ingest_one(data, PATH)

    assert (result.status, result.error_kind) == ('failed', 'incomplete_write')
    assert os_fake.items == {}
    assert set(os_fake.images) == {image_id}
    assert content_rows.rows == {content_imohash(data): image_id}, 'the doc exists: keep the claim'


@pytest.mark.asyncio
async def test_a_409_on_an_incomplete_old_doc_repairs_it() -> None:
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=14)
    image_id = _image_id(data)
    os_fake.images[image_id] = {
        'image_id': image_id,
        'image_path': '/elsewhere/x.jpg',
        'ingest_write_id': 'another-writer',
        'item_ids': ['their-item'],
    }
    _old(os_fake, image_id)

    result = await svc.ingest_one(data, PATH)

    assert result.status == 'success'
    assert len(os_fake.items) == 2
    assert sorted(os_fake.items) == os_fake.images[image_id]['item_ids']
    assert os_fake.images[image_id]['image_path'] == '/elsewhere/x.jpg'


@pytest.mark.asyncio
async def test_a_lookup_that_releases_a_live_claim_never_makes_a_second_image(
    content_rows: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claims and has not written; B's lookup finds the row without a doc and releases it
    (an orphan), then A writes. One images doc, whole; B answers duplicate or incomplete."""
    svc, os_fake, _ = _make_service(detections=TWO_BOXES)
    data = _jpeg_bytes(seed=15)
    a_claimed, b_looked_up = asyncio.Event(), asyncio.Event()
    real_bulk, real_resolve = os_fake.bulk, content_key.resolve_many_bound
    first_lookup = {'done': False}

    async def resolve(entries: Any) -> Any:
        found = await real_resolve(entries)
        if first_lookup['done']:
            b_looked_up.set()
        first_lookup['done'] = True
        return found

    async def bulk(*, body: list[dict[str, Any]], **kw: Any) -> Any:
        if not _is_items_write(body) and 'create' in body[0]:
            a_claimed.set()
            await asyncio.wait_for(b_looked_up.wait(), timeout=10)
        return await real_bulk(body=body, **kw)

    monkeypatch.setattr(content_key, 'resolve_many_bound', resolve)
    os_fake.bulk = bulk

    async def second() -> Any:
        await asyncio.wait_for(a_claimed.wait(), timeout=10)
        return await svc.ingest_one(data, '/b/copy.jpg')

    first_result, second_result = await asyncio.gather(
        svc.ingest_one(data, '/a/copy.jpg'), second()
    )

    assert first_result.status == 'success'
    assert second_result.status in ('duplicate', 'failed')
    if second_result.status == 'failed':
        assert second_result.error_kind == 'incomplete_write'
    assert len(os_fake.images) == 1
    assert sorted(os_fake.items) == os_fake.images[_image_id(data)]['item_ids']


@pytest.mark.asyncio
async def test_the_writer_lists_the_units_items_on_the_images_doc() -> None:
    os_fake = FakeIngestOpenSearch()
    writer = ingest_writer.IngestWriter(os_fake)
    cfg = get_curation_config()

    unit = await writer.write(
        {'image_id': 'img'},
        [{'crop_id': 'b', 'image_id': 'img'}, {'crop_id': 'a', 'image_id': 'img'}],
        images_index=cfg.images_index,
        items_index=cfg.items_index,
        guards=(),
    )

    assert unit.images_indexed == 1
    assert os_fake.images['img']['item_ids'] == ['a', 'b']
