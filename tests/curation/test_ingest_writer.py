"""The coalescing ingest writer: fewer requests, the same documents, one answer per image."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from curation.ingest_fakes import _jpeg_bytes, _make_service
from curation.occ_fakes import FakeIngestOpenSearch
from src.config import get_curation_config
from src.services.curation.ingest_writer import IngestWriter


GUARDS = ('label_source', 'class_source')


def _item(crop_id: str) -> dict[str, Any]:
    return {'crop_id': crop_id, 'image_id': crop_id[:2], 'class_name': 'widget'}


async def _write(writer: IngestWriter, image_id: str | None, crops: list[str], **kw: Any) -> Any:
    """``image_id`` None writes items only, onto an image that exists (a dataset import)."""
    cfg = get_curation_config()
    return await writer.write(
        {'image_id': image_id} if image_id else None,
        [_item(c) for c in crops],
        images_index=cfg.images_index,
        items_index=cfg.items_index,
        guards=GUARDS,
        **kw,
    )


@pytest.mark.asyncio
async def test_concurrent_images_share_one_set_of_requests() -> None:
    svc, os_fake, _ = _make_service()
    mgets: list[int] = []
    real_mget = os_fake.mget

    async def counting_mget(**kw: Any) -> Any:
        mgets.append(len(kw['body']['ids']))
        return await real_mget(**kw)

    os_fake.mget = counting_mget  # type: ignore[method-assign]
    results = await asyncio.gather(
        *(svc.ingest_one(_jpeg_bytes(seed=i), f'/tmp/p{i}.jpg') for i in range(8))
    )
    assert [r.status for r in results] == ['success'] * 8
    assert [r.crops_created for r in results] == [1] * 8
    assert len(os_fake.images) == 8
    assert len(os_fake.items) == 8
    # Per image the old path made one images bulk, one mget and one items bulk (24 requests).
    assert len(os_fake.bulk_calls) + len(mgets) <= 6


@pytest.mark.asyncio
async def test_a_refused_document_fails_only_its_own_image() -> None:
    os_fake = FakeIngestOpenSearch()
    real_bulk = os_fake.bulk

    async def refusing_bulk(*, body: Any, refresh: Any = False) -> Any:
        resp = await real_bulk(body=body, refresh=refresh)
        for item in resp['items']:
            if item.get('create', {}).get('_id') == 'bad-item':
                item['create']['status'] = 400
                item['create']['error'] = {'type': 'mapper_parsing_exception', 'reason': 'x'}
                os_fake.items.pop('bad-item')
        return resp

    os_fake.bulk = refusing_bulk  # type: ignore[method-assign]
    writer = IngestWriter(os_fake)
    good1, bad, good2 = await asyncio.gather(
        _write(writer, 'a', ['a1', 'a2']),
        _write(writer, 'b', ['bad-item', 'b2']),
        _write(writer, 'c', ['c1']),
    )
    assert writer.bulk_sizes == [8]
    assert (good1.created, good1.refused) == (2, [])
    assert (good2.created, good2.refused) == (1, [])
    assert bad.created == 1
    assert len(bad.refused) == 1
    assert 'bad-item' in bad.refused[0]
    assert 'mapper_parsing_exception' in bad.refused[0]
    assert set(os_fake.items) == {'a1', 'a2', 'b2', 'c1'}
    assert set(os_fake.images) == {'a', 'b', 'c'}


@pytest.mark.asyncio
async def test_rejected_items_are_retried_after_429() -> None:
    os_fake = FakeIngestOpenSearch()
    real_bulk = os_fake.bulk
    seen: list[str] = []

    async def queue_full_once(*, body: Any, refresh: Any = False) -> Any:
        ids = [a['create']['_id'] for a in body[0::2] if 'create' in a]
        if ids and not seen:
            seen.append('first')
            return {
                'errors': True,
                'items': [
                    {'create': {'_id': i, 'status': 429, 'error': {'type': 'rejected'}}}
                    for i in ids
                ],
            }
        return await real_bulk(body=body, refresh=refresh)

    os_fake.bulk = queue_full_once  # type: ignore[method-assign]
    writer = IngestWriter(os_fake)
    one, two = await asyncio.gather(_write(writer, 'a', ['a1']), _write(writer, 'b', ['b1']))
    assert (one.created, one.refused, two.created, two.refused) == (1, [], 1, [])
    assert set(os_fake.items) == {'a1', 'b1'}


@pytest.mark.asyncio
async def test_same_item_in_two_writes_never_shares_a_flush() -> None:
    os_fake = FakeIngestOpenSearch()
    writer = IngestWriter(os_fake)
    first, second = await asyncio.gather(
        _write(writer, 'a', ['same', 'a2']), _write(writer, None, ['same'])
    )
    assert writer.bulk_sizes == [3, 1]
    assert (first.created, second.created, second.updated) == (2, 0, 1)


@pytest.mark.asyncio
async def test_a_failed_flush_answers_every_caller() -> None:
    os_fake = FakeIngestOpenSearch()

    async def down(**_kw: Any) -> Any:
        msg = 'connection refused'
        raise ConnectionError(msg)

    os_fake.mget = down  # type: ignore[method-assign]
    writer = IngestWriter(os_fake)
    outcomes = await asyncio.wait_for(
        asyncio.gather(
            _write(writer, 'a', ['a1']), _write(writer, 'b', ['b1']), return_exceptions=True
        ),
        timeout=5,
    )
    assert all(isinstance(o, ConnectionError) for o in outcomes)
