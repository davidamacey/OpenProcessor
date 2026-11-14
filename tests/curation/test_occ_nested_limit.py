"""Every OCC write path refuses a box-list write that would reach the nested-document limit
(OpenSearch k-NN issue 3623), accepts one just below it, and writes nothing when it refuses."""

from __future__ import annotations

from typing import Any

import pytest

from curation.occ_fakes import VECTOR_EXCLUDES, FakeIngestOpenSearch, FakeOccOpenSearch
from src.clients.occ import occ_skip_on_conflict_bulk, occ_update_one
from src.clients.occ_bulk import occ_bulk_rounds, occ_update_bulk
from src.config import get_curation_config, get_region_fields
from src.domain.nested_limit import NESTED_DOCS_LIMIT, NestedLimitError


F = get_region_fields()
BELOW = 74
AT = 75


def _boxes(n: int) -> list[dict[str, Any]]:
    return [{'box_id': f'b{i}'} for i in range(n)]


def _write_boxes(n: int):
    def merge(_source: dict[str, Any]) -> dict[str, Any]:
        return {F.boxes: _boxes(n)}

    return merge


def _items_index() -> str:
    return get_curation_config().items_index


@pytest.mark.asyncio
async def test_occ_update_one_accepts_just_below_and_refuses_at_the_limit() -> None:
    client = FakeIngestOpenSearch(items={'c1': {F.boxes: []}, 'c2': {F.boxes: []}})
    await occ_update_one(
        client, doc_id='c1', merger=_write_boxes(BELOW), vector_excludes=VECTOR_EXCLUDES
    )
    assert len(client.items['c1'][F.boxes]) == BELOW

    with pytest.raises(NestedLimitError) as err:
        await occ_update_one(
            client, doc_id='c2', merger=_write_boxes(AT), vector_excludes=VECTOR_EXCLUDES
        )
    assert err.value.nested_docs == NESTED_DOCS_LIMIT
    assert client.items['c2'][F.boxes] == []
    assert [c['id'] for c in client.update_calls] == ['c1']


@pytest.mark.asyncio
async def test_occ_update_one_refuses_any_other_write_on_an_over_limit_item() -> None:
    client = FakeIngestOpenSearch(items={'c1': {F.boxes: _boxes(80)}})
    with pytest.raises(NestedLimitError):
        await occ_update_one(
            client,
            doc_id='c1',
            merger=lambda _s: {'class_id': 3},
            vector_excludes=VECTOR_EXCLUDES,
        )
    assert client.update_calls == []


@pytest.mark.asyncio
async def test_occ_update_one_repair_write_empties_the_box_vectors() -> None:
    client = FakeIngestOpenSearch(items={'c1': {F.boxes: _boxes(80)}})
    await occ_update_one(
        client, doc_id='c1', merger=_write_boxes(40), vector_excludes=VECTOR_EXCLUDES
    )
    assert len(client.items['c1'][F.boxes]) == 40
    assert client.items['c1'][F.box_embeddings] == []


@pytest.mark.asyncio
async def test_skip_on_conflict_bulk_refuses_the_item_and_writes_nothing_for_it() -> None:
    client = FakeOccOpenSearch({'ok': {F.boxes: []}, 'big': {F.boxes: []}})

    def merger(doc_id: str, _source: dict[str, Any]) -> dict[str, Any]:
        return {F.boxes: _boxes(BELOW if doc_id == 'ok' else AT)}

    result = await occ_skip_on_conflict_bulk(
        client,
        doc_ids=['ok', 'big'],
        merger=merger,
        vector_excludes=VECTOR_EXCLUDES,
        index=_items_index(),
    )
    assert result['updated_ids'] == ['ok']
    assert result['refused_over_nested_limit'] == 1
    assert result['errors'] == [
        {'doc_id': 'big', 'phase': 'guard', 'error': 'item_over_nested_limit'}
    ]
    written = [a['update']['_id'] for a in client.bulk_calls[0][0::2]]
    assert written == ['ok']


@pytest.mark.asyncio
async def test_occ_update_bulk_reports_nested_limit_per_item() -> None:
    client = FakeOccOpenSearch({'ok': {F.boxes: []}, 'big': {F.boxes: []}})
    status = await occ_update_bulk(
        client,
        ids=['ok', 'big'],
        merge_fn=lambda doc_id, _s: {F.boxes: _boxes(BELOW if doc_id == 'ok' else AT)},
        vector_excludes=VECTOR_EXCLUDES,
        index=_items_index(),
    )
    assert status == {'ok': 'updated', 'big': 'nested-limit'}
    assert [a['update']['_id'] for a in client.bulk_calls[0][0::2]] == ['ok']


@pytest.mark.asyncio
async def test_occ_bulk_rounds_fails_the_refused_item_with_the_nested_limit_reason() -> None:
    client = FakeOccOpenSearch({'big': {F.boxes: []}})
    failed: list[tuple[str, str]] = []
    await occ_bulk_rounds(
        client,
        index=_items_index(),
        ids=['big'],
        prepare=lambda _id, _doc: {F.boxes: _boxes(AT)},
        on_written=lambda _id: None,
        on_failed=lambda doc_id, _doc, why: failed.append((doc_id, why)),
        max_retries=0,
        source_excludes=VECTOR_EXCLUDES,
    )
    assert failed == [('big', 'nested_limit')]
    assert client.bulk_calls == []
