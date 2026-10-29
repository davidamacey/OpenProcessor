"""A transport retry of an images bulk that was already applied is not a duplicate (#375).

opensearch-py retries a request on a connection error or a 502/503/504. When the first
bulk was applied and only its response was lost, the retry's ``create`` of every images
doc answers 409. Those 409s answer this writer's own earlier write: the image must be
reported written and its items written, not reported ``duplicate`` with the items
dropped (which would leave an images doc with no items, and every later ingest of the
content would also say ``duplicate``). A 409 on a doc another writer stored stays a
duplicate with no items.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from curation.ingest_fakes import _jpeg_bytes, _make_service
from curation.occ_fakes import FakeIngestOpenSearch
from src.clients.curation_opensearch.bodies_core import _images_body
from src.config import get_curation_config
from src.services.curation.ingest_writer import WRITE_ID_FIELD, IngestWriter


def _lose_first_images_response(os_fake: FakeIngestOpenSearch) -> list[int]:
    """The first images bulk is applied, then its response is lost and the transport
    re-sends the same body, as opensearch-py does on a connection error."""
    real_bulk = os_fake.bulk
    retried: list[int] = []
    items_index = get_curation_config().items_index

    async def bulk(*, body: Any, refresh: Any = False) -> Any:
        images_bulk = any(
            a.get('create', {}).get('_index') not in (None, items_index) for a in body[0::2]
        )
        if images_bulk and not retried:
            await real_bulk(body=body, refresh=refresh)
            retried.append(len(body) // 2)
        return await real_bulk(body=body, refresh=refresh)

    os_fake.bulk = bulk  # type: ignore[method-assign]
    return retried


def _kw() -> dict[str, Any]:
    cfg = get_curation_config()
    return {'images_index': cfg.images_index, 'items_index': cfg.items_index, 'guards': ()}


@pytest.mark.asyncio
async def test_a_retried_images_bulk_writes_the_items_and_is_not_a_duplicate() -> None:
    os_fake = FakeIngestOpenSearch()
    retried = _lose_first_images_response(os_fake)
    writer = IngestWriter(os_fake)

    one, two = await asyncio.gather(
        writer.write({'image_id': 'img-a'}, [{'crop_id': 'a1', 'image_id': 'img-a'}], **_kw()),
        writer.write({'image_id': 'img-b'}, [{'crop_id': 'b1', 'image_id': 'img-b'}], **_kw()),
    )

    assert retried == [2]
    assert (one.duplicate, one.images_indexed, one.created, one.refused) == (False, 1, 1, [])
    assert (two.duplicate, two.images_indexed, two.created, two.refused) == (False, 1, 1, [])
    assert set(os_fake.images) == {'img-a', 'img-b'}
    assert set(os_fake.items) == {'a1', 'b1'}


@pytest.mark.asyncio
async def test_in_a_retried_bulk_a_doc_another_writer_stored_stays_a_duplicate() -> None:
    os_fake = FakeIngestOpenSearch(
        images={'theirs': {'image_id': 'theirs', WRITE_ID_FIELD: 'another-writer'}}
    )
    retried = _lose_first_images_response(os_fake)
    writer = IngestWriter(os_fake)

    mine, dup = await asyncio.gather(
        writer.write({'image_id': 'mine'}, [{'crop_id': 'm1', 'image_id': 'mine'}], **_kw()),
        writer.write({'image_id': 'theirs'}, [{'crop_id': 't1', 'image_id': 'theirs'}], **_kw()),
    )

    assert retried == [2]
    assert (mine.duplicate, mine.images_indexed, mine.created) == (False, 1, 1)
    assert (dup.duplicate, dup.images_indexed, dup.created, dup.refused) == (True, 0, 0, [])
    assert set(os_fake.items) == {'m1'}
    assert os_fake.images['theirs'][WRITE_ID_FIELD] == 'another-writer'


@pytest.mark.asyncio
async def test_an_ingest_whose_images_response_was_lost_succeeds() -> None:
    svc, os_fake, _ = _make_service()
    _lose_first_images_response(os_fake)

    result = await svc.ingest_one(_jpeg_bytes(seed=21), '/a/photo.jpg')

    assert result.status == 'success'
    assert result.crops_created == 1
    assert len(os_fake.images) == 1
    assert {d['image_id'] for d in os_fake.items.values()} == set(os_fake.images)


def test_the_write_id_is_mapped_stored_only() -> None:
    spec = _images_body()['mappings']['properties'][WRITE_ID_FIELD]
    assert spec == {'type': 'keyword', 'index': False, 'doc_values': False}


@pytest.mark.asyncio
async def test_a_refused_images_doc_gets_no_items() -> None:
    """Items written under an images doc the store refused would be orphans."""
    os_fake = FakeIngestOpenSearch()
    real_bulk = os_fake.bulk
    items_index = get_curation_config().items_index

    async def refuse_images(*, body: Any, refresh: Any = False) -> Any:
        if any(a.get('create', {}).get('_index') not in (None, items_index) for a in body[0::2]):
            return {
                'errors': True,
                'items': [
                    {'create': {'_id': a['create']['_id'], 'status': 400, 'error': 'mapper'}}
                    for a in body[0::2]
                ],
            }
        return await real_bulk(body=body, refresh=refresh)

    os_fake.bulk = refuse_images  # type: ignore[method-assign]
    result = await IngestWriter(os_fake).write(
        {'image_id': 'img'}, [{'crop_id': 'c1', 'image_id': 'img'}], **_kw()
    )

    assert (result.images_indexed, result.created, result.duplicate) == (0, 0, False)
    assert result.refused == ['images doc: 400: mapper']
    assert os_fake.items == {}
