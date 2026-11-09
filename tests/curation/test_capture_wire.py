"""The item `capture` wire block and the taken_from / taken_to filter (CM-2, plan 5.1 Wire)."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from _embedding_registry_fake import DEFAULT_STATE_FIELDS, DEFAULT_WRITE_FIELDS
from opensearchpy.exceptions import NotFoundError

from src.routers.curation._item_models import ItemDoc
from src.services.curation.capture_wire import CAPTURE_WIRE_FIELDS, capture_to_wire, image_capture
from src.services.curation.item_filter import ItemFilter, item_filter_clauses
from src.services.curation.wire import ITEM_WIRE_KEYS, serialize_item


STORED_BLOCK: dict[str, Any] = {
    'taken_local': '2024-03-05T14:22:11',
    'taken_utc': '2024-03-05T13:22:11',
    'camera_make': 'Acme',
    'has_location': True,
    'exif_status': 'ok',
    'parser': 'capture@1',
    'location': {'lat': 47.123456, 'lon': 8.987654},
    'altitude_m': 410.0,
    'location_precision': 'exact',
}


def test_the_wire_capture_is_a_whitelist_and_never_carries_a_position() -> None:
    wire = capture_to_wire(STORED_BLOCK)
    assert wire is not None
    assert set(wire) == set(CAPTURE_WIRE_FIELDS)
    assert wire['camera_make'] == 'Acme'
    assert wire['has_location'] is True
    assert '47.12' not in str(wire)
    assert not {'location', 'altitude_m', 'parser'} & set(wire)
    assert capture_to_wire(None) is None
    assert capture_to_wire({}) is None


def test_a_list_row_carries_the_two_times_and_a_row_without_them_carries_null() -> None:
    row = {'crop_id': 'c1', 'capture_taken_local': '2024-03-05T14:22:11'}
    wire = serialize_item(row, state_fields=DEFAULT_STATE_FIELDS)
    assert wire['capture']['taken_local'] == '2024-03-05T14:22:11'
    assert wire['capture']['taken_utc'] is None
    assert serialize_item({'crop_id': 'c2'}, state_fields=DEFAULT_STATE_FIELDS)['capture'] is None
    assert 'capture' in ITEM_WIRE_KEYS
    assert 'capture' in ItemDoc.model_fields


@pytest.mark.asyncio
async def test_the_detail_capture_comes_from_the_image_without_its_coordinates() -> None:
    class Client:
        async def get(self, **kw: Any) -> dict[str, Any]:
            assert kw['id'] == 'img1'
            assert kw['_source_includes'] == ['capture']
            return {'_source': {'capture': STORED_BLOCK}}

    got = await image_capture(Client(), 'img1')
    assert got is not None
    assert got['camera_make'] == 'Acme'
    assert 'location' not in got
    ItemDoc.model_validate(
        {**serialize_item({'crop_id': 'c'}, state_fields=DEFAULT_STATE_FIELDS), 'capture': got}
    )


@pytest.mark.asyncio
async def test_a_missing_image_or_block_leaves_the_item_as_it_is() -> None:
    class Gone:
        async def get(self, **_kw: Any) -> dict[str, Any]:
            raise NotFoundError(404, 'not_found', {})

    class Bare:
        async def get(self, **_kw: Any) -> dict[str, Any]:
            return {'_source': {}}

    assert await image_capture(Gone(), 'img1') is None
    assert await image_capture(Bare(), 'img1') is None
    assert await image_capture(Bare(), '') is None


def _clauses(**kw: Any) -> list[dict[str, Any]]:
    return item_filter_clauses(ItemFilter(**kw), embedding=DEFAULT_WRITE_FIELDS)


def test_taken_range_matches_utc_and_falls_back_to_local_when_there_is_no_utc() -> None:
    [clause] = _clauses(taken_from=dt.date(2024, 3, 5), taken_to=dt.date(2024, 3, 6))
    bounds = {'gte': '2024-03-05T00:00:00', 'lt': '2024-03-07T00:00:00'}
    utc, local = clause['bool']['should']
    assert utc == {'range': {'capture_taken_utc': bounds}}
    assert local['bool']['must'] == [{'range': {'capture_taken_local': bounds}}]
    assert local['bool']['must_not'] == [{'exists': {'field': 'capture_taken_utc'}}]


def test_a_one_sided_range_and_no_range() -> None:
    [clause] = _clauses(taken_to=dt.date(2024, 3, 5))
    assert clause['bool']['should'][0]['range']['capture_taken_utc'] == {
        'lt': '2024-03-06T00:00:00'
    }
    assert _clauses() == []


def test_an_inverted_range_is_refused() -> None:
    with pytest.raises(ValueError, match='taken_from'):
        _clauses(taken_from=dt.date(2024, 3, 6), taken_to=dt.date(2024, 3, 5))


def test_the_crops_route_declares_the_two_date_parameters() -> None:
    from src.routers.curation._item_filter_params import item_filter_query

    got = item_filter_query(taken_from=dt.date(2024, 1, 2), taken_to=None)
    assert got.taken_from == dt.date(2024, 1, 2)
