"""The consistency audit and the ``item_ids`` backfill, on a fake store.

``scripts/bench/ingest_consistency.py`` is the sweep's check that no photo is stored without its
objects (exit code 1 on any inconsistency); ``scripts/curation/backfill_item_ids.py`` stamps the
list on images docs written before the field existed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from scripts.bench import ingest_consistency as audit
from scripts.curation import backfill_item_ids as backfill
from src.config.curation_accessors import get_curation_config


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _indexes() -> tuple[str, str]:
    cfg = get_curation_config()
    return cfg.images_index, cfg.items_index


def _store(images: dict[str, dict[str, Any]], items: dict[str, str]) -> QueryFakeOpenSearch:
    images_index, items_index = _indexes()
    return QueryFakeOpenSearch(
        {
            images_index: images,
            items_index: {c: {'crop_id': c, 'image_id': i} for c, i in items.items()},
        }
    )


def _rows(*pages: list[str]) -> AsyncIterator[list[str]]:
    async def gen() -> AsyncIterator[list[str]]:
        for page in pages:
            yield page

    return gen()


@pytest.mark.asyncio
async def test_a_whole_project_has_no_inconsistency() -> None:
    images_index, items_index = _indexes()
    client = _store(
        {
            'a': {'image_id': 'a', 'item_ids': ['a1', 'a2']},
            'b': {'image_id': 'b', 'item_ids': []},
            'old': {'image_id': 'old'},
        },
        {'a1': 'a', 'a2': 'a', 'o1': 'old'},
    )

    report = await audit.audit(
        client, _rows(['a'], ['b', 'old']), images_index=images_index, items_index=items_index
    )

    assert report.inconsistencies == 0
    assert (report.images, report.items, report.content_rows) == (3, 3, 3)
    assert (report.images_without_list, report.images_with_empty_list) == (1, 1)


@pytest.mark.asyncio
async def test_each_kind_of_inconsistency_is_counted_with_an_example() -> None:
    images_index, items_index = _indexes()
    client = _store(
        {
            'a': {'image_id': 'a', 'item_ids': ['a1', 'a2']},  # a2 was never written
            'b': {'image_id': 'b', 'item_ids': ['b1']},
        },
        {'a1': 'a', 'b1': 'b', 'x1': 'ghost', 'x2': 'ghost'},  # two items of an image that is gone
    )

    report = await audit.audit(
        client,
        _rows(['a', 'b', 'claimed-only']),  # a content row whose images doc does not exist
        images_index=images_index,
        items_index=items_index,
    )

    assert report.images_missing_items == 1
    assert report.items_without_image == 2
    assert report.contents_without_image == 1
    assert report.inconsistencies == 4
    assert report.examples['images_missing_items'] == ['a']
    assert report.examples['contents_without_image'] == ['claimed-only']
    assert 'ghost' in report.examples['items_without_image'][0]


@pytest.mark.asyncio
async def test_the_audit_writes_nothing() -> None:
    images_index, items_index = _indexes()
    client = _store({'a': {'image_id': 'a', 'item_ids': ['gone']}}, {})

    await audit.audit(client, _rows(['a']), images_index=images_index, items_index=items_index)

    assert client.write_calls == 0


@pytest.mark.asyncio
async def test_the_backfill_stamps_legacy_images_and_leaves_the_rest() -> None:
    images_index, items_index = _indexes()
    client = _store(
        {
            'legacy': {'image_id': 'legacy'},
            'bare': {'image_id': 'bare'},
            'listed': {'image_id': 'listed', 'item_ids': ['l1']},
        },
        {'g1': 'legacy', 'g2': 'legacy', 'l1': 'listed', 'extra': 'listed'},
    )

    counts = await backfill.backfill(
        client, images_index=images_index, items_index=items_index, apply=True
    )

    docs = client.docs(images_index)
    assert docs['legacy']['item_ids'] == ['g1', 'g2']
    assert docs['bare']['item_ids'] == []
    assert docs['listed']['item_ids'] == ['l1'], 'a doc that has the field is left alone'
    assert (counts.scanned, counts.legacy, counts.stamped, counts.stamped_empty) == (3, 2, 2, 1)
    assert (counts.already_listed, counts.failed) == (1, 0)
    again = await backfill.backfill(
        client, images_index=images_index, items_index=items_index, apply=True
    )
    assert (again.legacy, again.stamped) == (0, 0), 'a second run finds nothing to do'


@pytest.mark.asyncio
async def test_the_backfill_dry_run_writes_nothing() -> None:
    images_index, items_index = _indexes()
    client = _store({'legacy': {'image_id': 'legacy'}}, {'g1': 'legacy'})

    counts = await backfill.backfill(
        client, images_index=images_index, items_index=items_index, apply=False
    )

    assert (counts.legacy, counts.stamped) == (1, 0)
    assert 'item_ids' not in client.docs(images_index)['legacy']
    assert client.write_calls == 0


@pytest.mark.asyncio
async def test_a_backfilled_project_passes_the_audit() -> None:
    images_index, items_index = _indexes()
    client = _store({'a': {'image_id': 'a'}, 'b': {'image_id': 'b'}}, {'a1': 'a'})

    await backfill.backfill(client, images_index=images_index, items_index=items_index, apply=True)
    report = await audit.audit(
        client, _rows(['a', 'b']), images_index=images_index, items_index=items_index
    )

    assert report.inconsistencies == 0
    assert report.images_without_list == 0


@pytest.mark.asyncio
async def test_max_images_bounds_a_dry_run_and_an_apply() -> None:
    images_index, items_index = _indexes()
    client = _store({f'i{n}': {'image_id': f'i{n}'} for n in range(5)}, {})

    dry = await backfill.backfill(
        client, images_index=images_index, items_index=items_index, apply=False, max_images=2
    )
    assert (dry.legacy, dry.stamped) == (2, 0)

    done = await backfill.backfill(
        client, images_index=images_index, items_index=items_index, apply=True, max_images=2
    )
    assert (done.legacy, done.stamped) == (2, 2)
    assert sum('item_ids' in d for d in client.docs(images_index).values()) == 2
