"""The gallery reads stored per-image counts for its sort and `label_state` filter (R-13).

Until every image in the list's scope has `counts_at`, the sort option is not served and
`label_state` keeps resolving through the items join; afterwards the filter is a range query on the
images index and the sort is served. The stored path holds only with no item narrowing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fixtures.fake_valkey import FakeValkeySets

from curation.query_fakes import QueryFakeOpenSearch
from curation.test_images_view import BASE, _client, _world
from src.clients.image_counts_dirty import DirtyImages
from src.config import IndexRole, index_name
from src.config.curation_accessors import base_curation_config
from src.services.curation import image_counts_refresh as refresh


CFG = base_curation_config()
ITEMS = CFG.items_index
IMAGES = index_name(CFG, IndexRole.IMAGES)
SORT = 'unlabeled_count:desc'


class _Recording(QueryFakeOpenSearch):
    """The world with every search recorded as ``(index, body)``."""

    def __init__(self, indexes: dict[str, dict[str, dict[str, Any]]]) -> None:
        super().__init__(indexes)
        self.searches: list[tuple[str, dict[str, Any]]] = []

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
        self.searches.append((index, body))
        return await super().search(index=index, body=body, **kw)

    def joins(self) -> list[dict[str, Any]]:
        """The item searches that resolve a predicate to images (a terms aggregation on image_id)."""
        return [
            b
            for i, b in self.searches
            if i == ITEMS
            and 'terms' in (b.get('aggs') or {}).get('images', {})
            and 'aggs' not in b['aggs']['images']
        ]


def _counted(counted: bool) -> _Recording:
    world = _world()
    fake = _Recording(world.store)
    if counted:
        asyncio.run(refresh.drain(fake, DirtyImages(FakeValkeySets())))
        fake.searches.clear()
    return fake


def _ids(body: dict[str, Any]) -> list[str]:
    return [c['image_id'] for c in body['items']]


def _get(fake: Any, monkeypatch: pytest.MonkeyPatch, **params: str) -> Any:
    return _client(fake, monkeypatch).get(f'{BASE}/images', params=params)


def test_unlabeled_sort_is_served_only_after_backfill(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _get(_counted(False), monkeypatch).json()
    assert SORT not in [o['id'] for o in before['sort_options']]
    refused = _get(_counted(False), monkeypatch, sort=SORT)
    assert refused.status_code == 422
    assert refused.json()['detail']['error'] == 'image_counts_not_ready'
    assert 'backfill' in refused.json()['detail']['message']

    after = _get(_counted(True), monkeypatch, sort=SORT)
    assert after.status_code == 200
    body = after.json()
    assert SORT in [o['id'] for o in body['sort_options']]
    assert 'unlabeled_count:asc' in [o['id'] for o in body['sort_options']]
    assert body['sort_applied'] == SORT
    assert _ids(body)[0] == 'img-a'  # the only image with an unlabeled item comes first


def test_one_uncounted_image_withdraws_the_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _counted(True)
    del fake.docs(IMAGES)['img-c']['counts_at']
    body = _get(fake, monkeypatch).json()
    assert SORT not in [o['id'] for o in body['sort_options']]


@pytest.mark.parametrize(
    ('state', 'expected'),
    [('has_unlabeled', ['img-a']), ('all_labeled', ['img-b', 'img-c']), ('no_items', ['img-d'])],
)
def test_label_state_reads_stored_counts_without_a_join(
    monkeypatch: pytest.MonkeyPatch, state: str, expected: list[str]
) -> None:
    fake = _counted(True)
    body = _get(fake, monkeypatch, label_state=state, sort='indexed_at:asc').json()
    assert sorted(_ids(body)) == expected
    assert fake.joins() == []
    # the page's images query, then only the cards' counts and the hidden counts on the items
    assert [i for i, _ in fake.searches if i == ITEMS] == [ITEMS, ITEMS]
    assert fake.searches[0][0] == IMAGES


@pytest.mark.parametrize('state', ['has_unlabeled', 'all_labeled', 'no_items'])
def test_the_join_and_the_stored_counts_agree(monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    joined = _counted(False)
    stored = _counted(True)
    a = _get(joined, monkeypatch, label_state=state, sort='indexed_at:asc').json()
    b = _get(stored, monkeypatch, label_state=state, sort='indexed_at:asc').json()
    assert joined.joins() != []
    assert stored.joins() == []
    assert _ids(a) == _ids(b)
    assert a['total'] == b['total']


def test_with_an_item_filter_label_state_keeps_the_join(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _counted(True)
    body = _get(
        fake, monkeypatch, label_state='has_unlabeled', class_name='sedan', sort='indexed_at:asc'
    ).json()
    assert fake.joins() != []
    assert isinstance(body['items'], list)


def test_with_held_out_items_shown_label_state_keeps_the_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _counted(True)
    _get(fake, monkeypatch, label_state='all_labeled', include_test='true', sort='indexed_at:asc')
    assert fake.joins() != []


def test_locate_works_under_the_stored_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _counted(True)
    out = _client(fake, monkeypatch).get(
        f'{BASE}/images/locate', params={'target_image_id': 'img-a', 'sort': SORT}
    )
    assert out.status_code == 200
    assert out.json()['in_view'] is True
    assert out.json()['index'] == 0
