"""The list an images doc keeps of its items stays true for every writer that removes items.

``item_ids`` names items that exist (``src/domain/image_completeness.py``). Every deletion goes
through :func:`~src.services.curation.item_delete.delete_items`, which takes the ids off the list
first, so an image whose items were removed on purpose never looks incomplete and a re-upload
never "repairs" it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from opensearchpy.exceptions import OpenSearchException

from curation.ingest_fakes import _jpeg_bytes, _make_service
from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import get_curation_config
from src.domain.image_completeness import listed_item_ids, repairable
from src.services.curation.image_completeness import forget_items, missing_items
from src.services.curation.item_delete import delete_items
from src.services.near_duplicates.content_key import content_image_id, content_imohash


def _items_index() -> str:
    return get_curation_config().items_index


def _images_index() -> str:
    return get_curation_config().images_index


async def _always(_crop_id: str, _doc: dict[str, Any]) -> bool:
    return True


def _world() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            _items_index(): {
                'c1': {'crop_id': 'c1', 'image_id': 'img-1'},
                'c2': {'crop_id': 'c2', 'image_id': 'img-1'},
                'c3': {'crop_id': 'c3', 'image_id': 'img-2'},
                'c4': {'crop_id': 'c4', 'image_id': 'img-3'},
            },
            _images_index(): {
                'img-1': {'image_id': 'img-1', 'item_ids': ['c1', 'c2']},
                'img-2': {'image_id': 'img-2', 'item_ids': ['c3', 'never-listed-by-others']},
                'img-3': {'image_id': 'img-3'},  # legacy: no list
            },
        }
    )


@pytest.mark.asyncio
async def test_a_deleted_item_leaves_the_list_of_its_image() -> None:
    fake = _world()

    result = await delete_items(
        fake, ['c1', 'c3', 'c4'], items_index=_items_index(), crop_cache_dir=None, deletable=_always
    )

    assert result['deleted'] == 3
    images = fake.docs(_images_index())
    assert images['img-1']['item_ids'] == ['c2']
    assert images['img-2']['item_ids'] == ['never-listed-by-others']
    assert 'item_ids' not in images['img-3'], 'a legacy image gets no list from a delete'
    assert set(fake.docs(_items_index())) == {'c2'}


@pytest.mark.asyncio
async def test_the_lists_of_many_images_are_updated_in_one_bulk_not_one_call_per_item() -> None:
    fake = _world()
    before = fake.bulk_calls

    await delete_items(
        fake,
        ['c1', 'c2', 'c3'],
        items_index=_items_index(),
        crop_cache_dir=None,
        deletable=_always,
    )

    assert fake.bulk_calls - before == 2, 'one bulk for the lists, one for the deletes'


@pytest.mark.asyncio
async def test_a_kept_item_stays_listed() -> None:
    fake = _world()

    async def not_c1(crop_id: str, _doc: dict[str, Any]) -> bool:
        return crop_id != 'c1'

    result = await delete_items(
        fake, ['c1', 'c2'], items_index=_items_index(), crop_cache_dir=None, deletable=not_c1
    )

    assert result['skipped'] == ['c1']
    assert fake.docs(_images_index())['img-1']['item_ids'] == ['c1']


@pytest.mark.asyncio
async def test_when_the_list_cannot_be_updated_nothing_is_deleted() -> None:
    fake = _world()
    real = fake.bulk

    async def refuse_lists(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        if body[0].get('update', {}).get('_index') == _images_index():
            raise OpenSearchException('images bulk failed')
        return await real(body=body, **kw)

    fake.bulk = refuse_lists

    result = await delete_items(
        fake, ['c1'], items_index=_items_index(), crop_cache_dir=None, deletable=_always
    )

    assert result['deleted'] == 0
    assert [e['crop_id'] for e in result['errors']] == ['c1']
    assert 'c1' in fake.docs(_items_index())


@pytest.mark.asyncio
async def test_forget_items_and_missing_items_read_the_lists() -> None:
    fake = _world()

    await forget_items(fake, _images_index(), {'img-1': {'c1'}, 'img-3': {'zzz'}})

    assert fake.docs(_images_index())['img-1']['item_ids'] == ['c2']
    assert await missing_items(
        fake, _items_index(), {'img-1': ['c2', 'gone'], 'img-2': ['c3']}
    ) == {'img-1': ['gone']}


@pytest.mark.asyncio
async def test_a_reprocess_that_removed_an_item_does_not_make_the_image_repairable() -> None:
    """The writer path end to end: ingest, then an item is deleted on purpose through
    ``delete_items`` (what a re-detection or an import undo does); a re-upload is a duplicate."""
    svc, os_fake, _ = _make_service(
        detections=[(0.05, 0.05, 0.4, 0.4, 0.9, 1), (0.5, 0.5, 0.9, 0.9, 0.9, 1)]
    )
    data = _jpeg_bytes(seed=31)
    await svc.ingest_one(data, '/p/x.jpg')
    image_id = content_image_id(content_imohash(data))
    victim = os_fake.images[image_id]['item_ids'][0]
    old = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    os_fake.images[image_id]['indexed_at'] = old

    await delete_items(
        os_fake, [victim], items_index=_items_index(), crop_cache_dir=None, deletable=_always
    )

    assert victim not in os_fake.items
    assert victim not in os_fake.images[image_id]['item_ids']
    assert (await svc.ingest_one(data, '/p/x.jpg')).status == 'duplicate'


def test_listed_item_ids_tells_legacy_from_empty() -> None:
    assert listed_item_ids({}) is None
    assert listed_item_ids({'item_ids': []}) == []
    assert listed_item_ids({'item_ids': ['a']}) == ['a']


def test_repairable_needs_a_mark_or_an_old_enough_doc() -> None:
    now = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    young = {'indexed_at': (now - timedelta(seconds=10)).isoformat()}
    old = {'indexed_at': (now - timedelta(seconds=5000)).isoformat()}
    assert not repairable(young, now=now, repair_after_s=100)
    assert repairable(old, now=now, repair_after_s=100)
    assert repairable({**young, 'ingest_incomplete': True}, now=now, repair_after_s=100)
    assert not repairable({**young, 'ingest_incomplete': False}, now=now, repair_after_s=100)
    # a date that cannot be read counts as young: never repair on a guess
    assert not repairable({'indexed_at': 'not a date'}, now=now, repair_after_s=100)
    assert not repairable({}, now=now, repair_after_s=100)
    # a naive timestamp is read as UTC
    assert repairable({'indexed_at': '2020-01-01T00:00:00'}, now=now, repair_after_s=100)


@pytest.mark.asyncio
async def test_a_batch_answers_each_photo_by_its_stored_image() -> None:
    """One request: a whole stored image, an incomplete one nobody writes, one a writer may still
    be writing, and a new one. Each gets its own answer; only the second is ingested again."""
    svc, os_fake, _ = _make_service()
    whole, repair, young, fresh = (_jpeg_bytes(seed=40 + i) for i in range(4))
    for data in (whole, repair, young):
        await svc.ingest_one(data, f'/p/{content_imohash(data)[:6]}.jpg')
    ids = {
        name: content_image_id(content_imohash(d))
        for name, d in (('repair', repair), ('young', young))
    }
    for name in ('repair', 'young'):
        for item_id in os_fake.images[ids[name]]['item_ids']:
            del os_fake.items[item_id]
    os_fake.images[ids['repair']]['indexed_at'] = (
        datetime.now(UTC) - timedelta(days=2)
    ).isoformat()

    batch = await svc.ingest_batch(
        [whole, repair, young, fresh], ['/a/1.jpg', '/a/2.jpg', '/a/3.jpg', '/a/4.jpg']
    )

    assert [(r.status, r.error_kind) for r in batch.results] == [
        ('duplicate', None),
        ('success', None),
        ('failed', 'incomplete_write'),
        ('success', None),
    ]
    assert batch.status == 'partial'
    assert (batch.summary.successful, batch.summary.duplicates, batch.summary.failed) == (2, 1, 1)
    assert os_fake.images[ids['repair']]['item_ids'] == sorted(
        i for i in os_fake.items if os_fake.items[i]['image_id'] == ids['repair']
    )
