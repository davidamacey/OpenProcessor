"""Every region row carries a non-empty ``row_key`` that is unique in a page,
including item-level rows (``region_box_id`` null) and items matched on
several boxes. ``crop_id`` alone repeats across the per-box rows of one item
by design; ``row_key`` is the list key."""

from __future__ import annotations

import pytest

from src.services.curation import region_rows as region_rows_mod
from src.services.curation.region_rows import INNER_HITS_NAME, as_row, region_rows, row_key


@pytest.fixture(autouse=True)
def _plain_items(monkeypatch: pytest.MonkeyPatch) -> None:
    # The wire serializer is covered elsewhere; only the row keys matter here.
    monkeypatch.setattr(
        region_rows_mod, 'serialize_item', lambda src, cid: {'crop_id': src.get('crop_id') or cid}
    )


def _hit(crop_id: str, box_ids: list[str], matched_offsets: list[int]) -> dict:
    return {
        '_id': crop_id,
        '_source': {'crop_id': crop_id, 'region_boxes': [{'box_id': b} for b in box_ids]},
        'inner_hits': {
            INNER_HITS_NAME: {
                'hits': {'hits': [{'_nested': {'offset': o}} for o in matched_offsets]}
            }
        },
    }


def test_item_level_rows_get_a_synthetic_unique_key_and_keep_null_box_id() -> None:
    rows = region_rows([_hit('a', ['b1'], []), _hit('c', [], [])], boxes_selected=False)
    assert [r['region_box_id'] for r in rows] == [None, None]
    keys = [r['row_key'] for r in rows]
    assert keys == ['a#item', 'c#item']


def test_per_box_rows_key_on_crop_and_box() -> None:
    rows = region_rows([_hit('a', ['b1', 'b2'], [0, 1])], boxes_selected=True)
    assert [r['row_key'] for r in rows] == ['a#b1', 'a#b2']
    assert {r['crop_id'] for r in rows} == {'a'}  # crop_id repeats; row_key does not


def test_a_box_id_repeated_inside_one_item_yields_one_row() -> None:
    rows = region_rows([_hit('a', ['b1', 'b1'], [0, 1])], boxes_selected=True)
    assert [r['row_key'] for r in rows] == ['a#b1']


def test_keys_are_never_empty_and_unique_across_a_mixed_page() -> None:
    hits = [_hit(f'c{i}', [f'x{i}', f'y{i}'], [0, 1]) for i in range(5)]
    keys = [r['row_key'] for r in region_rows(hits, boxes_selected=True)]
    assert all(keys)
    assert len(keys) == len(set(keys)) == 10


def test_as_row_key_is_the_one_shared_rule() -> None:
    assert as_row({'crop_id': 'k'}, 'bx')['row_key'] == row_key('k', 'bx') == 'k#bx'
    assert as_row({'crop_id': 'k'}, None)['row_key'] == row_key('k', None) == 'k#item'
