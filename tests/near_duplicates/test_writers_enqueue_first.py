"""A writer marks its members pending before the OpenSearch write it describes."""

from __future__ import annotations

from typing import Any

import pytest
from curation.query_fakes import QueryFakeOpenSearch

from src.config.curation_accessors import get_curation_config
from src.services.curation.item_delete import delete_items


async def _always(_crop_id: str, _doc: dict[str, Any]) -> bool:
    return True


@pytest.mark.asyncio
async def test_a_delete_is_marked_before_the_bulk_delete(pending_calls: Any) -> None:
    index = get_curation_config().items_index
    fake = QueryFakeOpenSearch({index: {'c1': {'crop_id': 'c1'}, 'c2': {'crop_id': 'c2'}}})
    real = fake.bulk
    seen_at_bulk: list[int] = []

    async def bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        seen_at_bulk.append(len(pending_calls.calls))
        return await real(body=body, **kw)

    fake.bulk = bulk  # type: ignore[method-assign]

    result = await delete_items(
        fake, ['c1', 'c2'], items_index=index, crop_cache_dir=None, deletable=_always
    )

    assert result['deleted'] == 2
    assert seen_at_bulk == [1]
    assert pending_calls.calls == [('item', ['c1', 'c2'], 'deleted')]


@pytest.mark.asyncio
async def test_an_images_index_delete_is_an_image_level_row(pending_calls: Any) -> None:
    index = get_curation_config().images_index
    fake = QueryFakeOpenSearch({index: {'i1': {'image_id': 'i1'}}})
    await delete_items(fake, ['i1'], items_index=index, crop_cache_dir=None, deletable=_always)
    assert pending_calls.calls == [('image', ['i1'], 'deleted')]


@pytest.mark.asyncio
async def test_a_foreign_index_is_refused_and_nothing_is_deleted(pending_calls: Any) -> None:
    fake = QueryFakeOpenSearch({'someone_elses_items': {'c1': {'crop_id': 'c1'}}})
    result = await delete_items(
        fake, ['c1'], items_index='someone_elses_items', crop_cache_dir=None, deletable=_always
    )
    assert set(fake.docs('someone_elses_items')) == {'c1'}
    assert result['deleted'] == 0
    assert len(result['errors']) == 1
    assert 'not an images or items index' in result['errors'][0]['error']
    assert pending_calls.calls == []


@pytest.mark.asyncio
async def test_the_region_worker_marks_changed_boxes_before_its_bulk_write(
    pending_calls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.curation.worker import bulk_writer
    from scripts.curation.worker.state import _ItemTask

    seen_at_write: list[int] = []

    async def fake_bulk(*_a: Any, **_kw: Any) -> dict[str, Any]:
        seen_at_write.append(len(pending_calls.calls))
        return {'updated': 0}

    monkeypatch.setattr(bulk_writer, 'occ_skip_on_conflict_bulk', fake_bulk)
    monkeypatch.setattr(bulk_writer, '_current_config_stamp', lambda: (None, None, None))

    def task(crop_id: str, **kw: Any) -> _ItemTask:
        return _ItemTask(
            crop_id=crop_id,
            image_path='/x.jpg',
            item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
            region_status='pending_detection',
            class_name='car',
            **kw,
        )

    boxes = task('moved', update_doc={'a': 1}, pending_boxes=[])
    status_only = task('status', update_doc={'a': 1})
    await bulk_writer._bulk_update_one_project(None, [boxes, status_only])  # type: ignore[arg-type]

    assert seen_at_write == [1]
    assert pending_calls.calls == [('item', ['moved'], 'boxes_changed')]
