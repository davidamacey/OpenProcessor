"""Coverage for ``src/services/curation/clustering/outliers.py`` (a
zero-coverage leaf feeding ``/curation/review/outliers``).
"""

from __future__ import annotations

from typing import Any

import pytest

import src.services.curation.clustering.outliers as outliers_mod
from src.services.curation.clustering.outliers import compute_outlier_order, make_cache_key


class _FakeScrollOS:
    """Minimal AsyncOpenSearch double for one scroll page (tests keep
    each cluster well under ``_SCROLL_PAGE``, so a single page suffices)."""

    def __init__(self, docs: dict[str, list[float]]) -> None:
        self._docs = docs

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        field = body['_source'][0]
        hits = [{'_id': doc_id, '_source': {field: emb}} for doc_id, emb in self._docs.items()]
        return {'_scroll_id': 'scroll-1', 'hits': {'hits': hits}}

    async def scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        return {'count': len(self._docs)}


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    outliers_mod._CACHE.clear()


def test_make_cache_key_is_stable_regardless_of_query_key_order() -> None:
    k1 = make_cache_key('idx', {'a': 1, 'b': 2}, 'pe_embedding')
    k2 = make_cache_key('idx', {'b': 2, 'a': 1}, 'pe_embedding')
    assert k1 == k2


def test_make_cache_key_differs_across_index_field_or_query() -> None:
    base = make_cache_key('idx', {'a': 1}, 'pe_embedding')
    assert make_cache_key('other', {'a': 1}, 'pe_embedding') != base
    assert make_cache_key('idx', {'a': 2}, 'pe_embedding') != base
    assert make_cache_key('idx', {'a': 1}, 'other_field') != base


@pytest.mark.asyncio
async def test_planted_far_member_ranks_first() -> None:
    # Four members tightly clustered around [1, 0, 0], one planted far
    # away near [0, 1, 0] — cosine-farthest from the centroid.
    docs = {
        'near-1': [1.0, 0.01, 0.0],
        'near-2': [1.0, -0.01, 0.0],
        'near-3': [0.99, 0.0, 0.01],
        'far-outlier': [0.0, 1.0, 0.0],
    }
    client = _FakeScrollOS(docs)
    order = await compute_outlier_order(client, 'op_items', {'term': {'cluster_id': 1}})
    assert order is not None
    assert order[0] == 'far-outlier'
    assert set(order) == set(docs)


@pytest.mark.asyncio
async def test_single_member_cluster_returns_that_one_id() -> None:
    client = _FakeScrollOS({'only-one': [1.0, 0.0, 0.0]})
    order = await compute_outlier_order(client, 'op_items', {'term': {'cluster_id': 2}})
    assert order == ['only-one']


@pytest.mark.asyncio
async def test_empty_cluster_returns_empty_list() -> None:
    client = _FakeScrollOS({})
    order = await compute_outlier_order(client, 'op_items', {'term': {'cluster_id': 3}})
    assert order == []


@pytest.mark.asyncio
async def test_too_large_cluster_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(outliers_mod, '_MAX_MEMBERS', 2)
    docs = {f'm{i}': [1.0, float(i) * 0.01, 0.0] for i in range(5)}
    client = _FakeScrollOS(docs)
    order = await compute_outlier_order(client, 'op_items', {'term': {'cluster_id': 4}})
    assert order is None


@pytest.mark.asyncio
async def test_cached_order_reused_when_count_unchanged() -> None:
    docs = {'a': [1.0, 0.0, 0.0], 'b': [0.0, 1.0, 0.0]}
    client = _FakeScrollOS(docs)
    query = {'term': {'cluster_id': 5}}

    first = await compute_outlier_order(client, 'op_items', query, current_count=2)

    # A client that would raise if actually queried again — proves the
    # second call served from cache rather than re-scrolling.
    class _BoomOS:
        async def search(self, **_kw: Any) -> dict[str, Any]:
            raise AssertionError('should not re-query while cache is valid')

    second = await compute_outlier_order(_BoomOS(), 'op_items', query, current_count=2)
    assert second == first


@pytest.mark.asyncio
async def test_cache_invalidated_when_member_count_changes() -> None:
    docs = {'a': [1.0, 0.0, 0.0], 'b': [0.0, 1.0, 0.0]}
    client = _FakeScrollOS(docs)
    query = {'term': {'cluster_id': 6}}

    await compute_outlier_order(client, 'op_items', query, current_count=2)

    docs3 = {**docs, 'c': [0.0, 0.0, 1.0]}
    client2 = _FakeScrollOS(docs3)
    second = await compute_outlier_order(client2, 'op_items', query, current_count=3)
    assert second is not None
    assert set(second) == set(docs3)


@pytest.mark.asyncio
async def test_members_missing_embedding_field_are_skipped() -> None:
    class _PartialOS:
        async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
            return {
                '_scroll_id': 's1',
                'hits': {
                    'hits': [
                        {'_id': 'no-embedding', '_source': {}},
                        {'_id': 'has-embedding', '_source': {'pe_embedding': [1.0, 0.0, 0.0]}},
                    ]
                },
            }

        async def scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
            return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

        async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
            return {'count': 2}

        async def clear_scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
            return {}

    order = await compute_outlier_order(_PartialOS(), 'op_items', {'term': {'cluster_id': 7}})
    assert order == ['has-embedding']


@pytest.mark.asyncio
async def test_too_large_cluster_never_issues_a_search_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count-first check must skip the scroll entirely (not just
    break out of it early) when the pool is already known to be too large."""
    monkeypatch.setattr(outliers_mod, '_MAX_MEMBERS', 2)

    class _CountOnlyOS:
        def __init__(self) -> None:
            self.search_calls = 0

        async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
            return {'count': 500}

        async def search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
            self.search_calls += 1
            raise AssertionError('must not scroll a pool already known to exceed _MAX_MEMBERS')

    client = _CountOnlyOS()
    order = await compute_outlier_order(client, 'op_items', {'term': {'cluster_id': 9}})
    assert order is None
    assert client.search_calls == 0
