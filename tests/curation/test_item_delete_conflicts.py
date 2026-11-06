"""``delete_items`` under a version conflict: a write that lands between the
re-read and the delete makes the delete re-read, never a silent no-op."""

from __future__ import annotations

from typing import Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import get_curation_config
from src.services.curation.item_delete import delete_items


def _index() -> str:
    """The bound project's items index (a delete marks near-duplicate rows by index role)."""
    return get_curation_config().items_index


async def _always(_crop_id: str, _doc: dict[str, Any]) -> bool:
    return True


def _store() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch({_index(): {'c1': {'crop_id': 'c1'}, 'c2': {'crop_id': 'c2'}}})


def _bump_before_each_bulk(fake: QueryFakeOpenSearch, victim: str, times: int) -> None:
    """A version-only write (the document is unchanged) just before the
    first ``times`` conditional bulk deletes."""
    real = fake.bulk
    left = {'n': times}

    async def bumping(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        if left['n'] > 0:
            left['n'] -= 1
            fake._bump(_index(), victim)
        return await real(body=body, **kw)

    fake.bulk = bumping  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_a_conflicted_delete_is_re_read_and_deleted_on_the_next_round() -> None:
    fake = _store()
    _bump_before_each_bulk(fake, 'c1', times=1)

    result = await delete_items(
        fake, ['c1', 'c2'], items_index=_index(), crop_cache_dir=None, deletable=_always
    )

    assert fake.docs(_index()) == {}
    assert (result['deleted'], result['skipped'], result['errors']) == (2, [], [])


@pytest.mark.asyncio
async def test_a_document_that_conflicts_every_round_is_reported_skipped_and_kept() -> None:
    fake = _store()
    _bump_before_each_bulk(fake, 'c1', times=99)

    result = await delete_items(
        fake, ['c1', 'c2'], items_index=_index(), crop_cache_dir=None, deletable=_always
    )

    assert set(fake.docs(_index())) == {'c1'}
    assert (result['deleted'], result['skipped'], result['errors']) == (1, ['c1'], [])
