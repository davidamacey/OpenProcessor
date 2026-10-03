"""``ItemSelection``: ids or a filter, capped by the largest boxes or a seeded sample."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from curation.query_fakes import QueryFakeOpenSearch
from src.services.curation import item_selection as sel_mod
from src.services.curation.item_filter import ItemFilter
from src.services.curation.item_selection import ItemSelection, SelectionError, resolve_selection


INDEX = 'items'


def _fake() -> QueryFakeOpenSearch:
    docs = {
        f'c{i:02d}': {
            'crop_id': f'c{i:02d}',
            'proposal_name': 'car' if i % 2 else 'dog',
            'crop_area_norm': i / 100,
            **({'test_holdout': True} if i == 7 else {}),
        }
        for i in range(1, 21)
    }
    return QueryFakeOpenSearch({INDEX: docs})


@pytest.mark.parametrize(
    'body',
    [
        {},
        {'crop_ids': ['a'], 'filter': {}},
        {'crop_ids': ['a'], 'limit': 3},
        {'filter': {'class_names': ['car']}, 'sample': 'random'},
        {'filter': {}},
        {'filter': {'bogus': 1}},
    ],
)
def test_malformed_selections_are_refused(body: dict) -> None:
    with pytest.raises(ValidationError):
        ItemSelection(**body)


@pytest.mark.asyncio
async def test_explicit_ids_are_deduplicated_and_keep_order() -> None:
    sel = ItemSelection(crop_ids=['b', 'a', 'b'])
    assert await resolve_selection(_fake(), sel, index=INDEX) == ['b', 'a']


@pytest.mark.asyncio
async def test_filter_selection_hides_holdout_unless_included() -> None:
    sel = ItemSelection(filter=ItemFilter(class_names=['car']))
    ids = await resolve_selection(_fake(), sel, index=INDEX)
    assert ids == ['c01', 'c03', 'c05', 'c09', 'c11', 'c13', 'c15', 'c17', 'c19']  # c07 held out
    with_test = ItemSelection(filter=ItemFilter(class_names=['car']), include_test=True)
    assert 'c07' in await resolve_selection(_fake(), with_test, index=INDEX)


@pytest.mark.asyncio
async def test_largest_cap_takes_the_biggest_boxes() -> None:
    sel = ItemSelection(filter=ItemFilter(class_names=['dog']), limit=3, sample='largest')
    assert await resolve_selection(_fake(), sel, index=INDEX) == ['c20', 'c18', 'c16']


@pytest.mark.asyncio
async def test_random_sample_is_seeded_and_bounded() -> None:
    def make(seed: int) -> ItemSelection:
        return ItemSelection(
            filter=ItemFilter(class_names=['car']), limit=4, sample='random', seed=seed
        )

    a = await resolve_selection(_fake(), make(1), index=INDEX)
    assert a == await resolve_selection(_fake(), make(1), index=INDEX)
    assert len(a) == len(set(a)) == 4
    others = {tuple(await resolve_selection(_fake(), make(n), index=INDEX)) for n in range(2, 8)}
    assert others != {tuple(a)}  # the seed changes the draw


@pytest.mark.asyncio
async def test_a_limit_above_the_match_count_returns_all_matches() -> None:
    sel = ItemSelection(filter=ItemFilter(class_names=['dog']), limit=100, sample='random')
    assert len(await resolve_selection(_fake(), sel, index=INDEX)) == 10


@pytest.mark.asyncio
async def test_a_bad_band_is_a_selection_error() -> None:
    sel = ItemSelection(filter=ItemFilter(conf_min=0.9, conf_max=0.1))
    with pytest.raises(SelectionError):
        await resolve_selection(_fake(), sel, index=INDEX)


@pytest.mark.asyncio
async def test_too_many_matches_to_resolve_is_refused_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sel_mod, 'MAX_SELECTION_SCAN', 5)
    sel = ItemSelection(filter=ItemFilter(class_names=['car', 'dog']), limit=2, sample='random')
    with pytest.raises(SelectionError, match='narrow'):
        await resolve_selection(_fake(), sel, index=INDEX)
