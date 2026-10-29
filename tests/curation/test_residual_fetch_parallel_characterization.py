"""Characterization of ``fetch_residual_embeddings_parallel`` (PIT + sliced fetch).

Pins the request bodies it sends (pool gate, slice and search_after paging,
PIT keep-alive), the order and normalization of what it returns, the
progress protocol and every degraded path: PIT unavailable (falls back to the
scroll fetch), a PIT response with no id, a failed pre-count, a failed PIT
delete, an empty pool and a cancel raised from progress.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.services.curation.clustering import residual_fetch
from src.services.curation.clustering.residual_fetch import (
    CONFIDENT_CLASS_SOURCES,
    RESIDUAL_EMBEDDING_FIELD,
    fetch_residual_embeddings_parallel,
)


class _CancelledError(Exception):
    pass


class _Progress:
    def __init__(self, cancel_after: int | None = None) -> None:
        self.updates: list[tuple[int, int]] = []
        self.cancel_checks = 0
        self._cancel_after = cancel_after

    def update(self, *, processed: int, total: int) -> None:
        self.updates.append((processed, total))

    def raise_if_cancelled(self) -> None:
        self.cancel_checks += 1
        if self._cancel_after is not None and self.cancel_checks >= self._cancel_after:
            raise _CancelledError


class _Client:
    """A PIT-capable fake over ``n`` documents; document ``i`` is in slice ``i % max``."""

    def __init__(
        self,
        n: int = 7,
        *,
        pit: Any = 'pit-1',
        fail_count: bool = False,
        fail_create_pit: bool = False,
        fail_delete_pit: bool = False,
        no_sort_key: bool = False,
    ) -> None:
        self.docs = [(f'd{i}', [float(i + 1), 2.0, float(i % 3)]) for i in range(n)]
        self.pit = pit
        self.fail_count = fail_count
        self.fail_create_pit = fail_create_pit
        self.fail_delete_pit = fail_delete_pit
        self.no_sort_key = no_sort_key
        self.counts: list[dict[str, Any]] = []
        self.bodies: list[dict[str, Any]] = []
        self.scroll_searches: list[dict[str, Any]] = []
        self.pit_requests: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []

    def _hit(self, i: int) -> dict[str, Any]:
        h: dict[str, Any] = {
            '_id': self.docs[i][0],
            '_source': {RESIDUAL_EMBEDDING_FIELD: self.docs[i][1]},
        }
        if not self.no_sort_key:
            h['sort'] = [i]
        return h

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.counts.append({'index': index, 'body': body})
        if self.fail_count:
            raise RuntimeError('count boom')
        return {'count': len(self.docs)}

    async def create_pit(self, *, index: str, keep_alive: str) -> dict[str, Any]:
        self.pit_requests.append({'index': index, 'keep_alive': keep_alive})
        if self.fail_create_pit:
            raise RuntimeError('pit boom')
        return {'pit_id': self.pit} if self.pit is not None else {}

    async def delete_pit(self, *, body: dict[str, Any]) -> None:
        self.deleted.append(body)
        if self.fail_delete_pit:
            from opensearchpy.exceptions import OpenSearchException

            raise OpenSearchException('delete boom')

    async def search(
        self, *, body: dict[str, Any], index: str | None = None, scroll: str | None = None
    ):
        if 'pit' not in body:  # the scroll fallback
            self.scroll_searches.append({'index': index, 'scroll': scroll, 'body': body})
            hits = [self._hit(i) for i in range(len(self.docs))]
            return {'_scroll_id': 'sc', 'hits': {'hits': hits}}
        self.bodies.append(body)
        sl = body.get('slice')
        after = body.get('search_after')
        idx = [i for i in range(len(self.docs)) if sl is None or i % sl['max'] == sl['id']]
        if after is not None:
            idx = [i for i in idx if i > after[0]]
        return {'hits': {'hits': [self._hit(i) for i in idx[: body['size']]]}}

    async def scroll(self, *, scroll_id: str, scroll: str) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str) -> None:  # noqa: ARG002
        return None


def _unit(rows: list[list[float]]) -> np.ndarray:
    m = np.asarray(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


@pytest.mark.asyncio
async def test_slices_page_with_search_after_and_return_normalized_rows_in_id_order() -> None:
    client = _Client(7)
    ids, matrix = await fetch_residual_embeddings_parallel(client, n_slices=3, page_size=2)
    # slice 0 = d0,d3,d6; slice 1 = d1,d4; slice 2 = d2,d5 -- merged in document-id order (#372)
    assert ids == ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6']
    order = [0, 1, 2, 3, 4, 5, 6]
    assert np.allclose(matrix, _unit([client.docs[i][1] for i in order]))
    assert matrix.dtype == np.float32
    per_slice: dict[int, list[Any]] = {}
    for b in client.bodies:
        per_slice.setdefault(b['slice']['id'], []).append((b.get('search_after'), b['size']))
    assert per_slice == {
        0: [(None, 2), ([3], 2), ([6], 2)],
        1: [(None, 2), ([4], 2)],
        2: [(None, 2), ([5], 2)],
    }
    body = client.bodies[0]
    assert body['pit'] == {'id': 'pit-1', 'keep_alive': '5m'}
    assert body['sort'] == [{'_shard_doc': 'asc'}]
    assert body['_source'] == [RESIDUAL_EMBEDDING_FIELD]
    assert body['slice'] == {'id': 0, 'max': 3}
    assert client.pit_requests == [{'index': client.pit_requests[0]['index'], 'keep_alive': '5m'}]
    assert client.deleted == [{'pit_id': ['pit-1']}]


@pytest.mark.asyncio
async def test_a_single_slice_sends_no_slice_clause() -> None:
    client = _Client(3)
    ids, _ = await fetch_residual_embeddings_parallel(client, n_slices=1, page_size=10)
    assert ids == ['d0', 'd1', 'd2']
    assert all('slice' not in b for b in client.bodies)


@pytest.mark.asyncio
async def test_a_hit_without_a_sort_key_ends_that_slice() -> None:
    client = _Client(5, no_sort_key=True)
    ids, _ = await fetch_residual_embeddings_parallel(client, n_slices=1, page_size=2)
    assert ids == ['d0', 'd1']
    assert len(client.bodies) == 1


@pytest.mark.asyncio
async def test_the_pool_gate_is_in_the_query() -> None:
    client = _Client(2)
    await fetch_residual_embeddings_parallel(
        client,
        include_candidate_clusters=False,
        candidate_cluster_id_min=100,
        extra_must=[{'range': {'rank': {'gte': 1}}}],
        n_slices=1,
    )
    query = client.bodies[0]['query']['bool']
    assert query['must_not'] == [
        {'term': {'class_validated': True}},
        {'terms': {'class_source': list(CONFIDENT_CLASS_SOURCES)}},
        {'term': {'class_excluded': True}},
    ]
    assert query['filter'][0] == residual_fetch.embedded_clause(RESIDUAL_EMBEDDING_FIELD)
    assert query['filter'][1] == {
        'bool': {
            'should': [
                {'bool': {'must_not': [{'exists': {'field': 'cluster_id'}}]}},
                {'range': {'cluster_id': {'lt': 100}}},
            ],
            'minimum_should_match': 1,
        }
    }
    assert query['filter'][2] == {'range': {'rank': {'gte': 1}}}
    assert len(query['filter']) == 3


@pytest.mark.asyncio
async def test_candidate_clusters_included_or_no_threshold_adds_no_cluster_clause() -> None:
    for kwargs in ({'include_candidate_clusters': True, 'candidate_cluster_id_min': 100}, {}):
        client = _Client(2)
        await fetch_residual_embeddings_parallel(client, n_slices=1, **kwargs)
        assert len(client.bodies[0]['query']['bool']['filter']) == 1


@pytest.mark.asyncio
async def test_progress_gets_a_precount_a_running_total_and_cancel_checks() -> None:
    client = _Client(5)
    progress = _Progress()
    await fetch_residual_embeddings_parallel(client, n_slices=1, page_size=2, progress=progress)
    assert client.counts[0]['body'] == {'query': client.bodies[0]['query']}
    assert progress.updates == [(0, 5), (2, 5), (4, 5), (5, 5)]
    assert progress.cancel_checks == 3


@pytest.mark.asyncio
async def test_no_progress_means_no_count_query() -> None:
    client = _Client(2)
    await fetch_residual_embeddings_parallel(client, n_slices=1)
    assert client.counts == []


@pytest.mark.asyncio
async def test_a_failed_precount_does_not_stop_the_fetch_and_total_falls_back_to_running_count() -> (
    None
):
    client = _Client(3, fail_count=True)
    progress = _Progress()
    ids, _ = await fetch_residual_embeddings_parallel(
        client, n_slices=1, page_size=10, progress=progress
    )
    assert ids == ['d0', 'd1', 'd2']
    assert progress.updates == [(3, 3)]


@pytest.mark.asyncio
async def test_a_cancel_raised_by_progress_still_deletes_the_pit() -> None:
    client = _Client(6)
    with pytest.raises(_CancelledError):
        await fetch_residual_embeddings_parallel(
            client, n_slices=1, page_size=2, progress=_Progress(cancel_after=2)
        )
    assert client.deleted == [{'pit_id': ['pit-1']}]


@pytest.mark.asyncio
@pytest.mark.parametrize('kwargs', [{'fail_create_pit': True}, {'pit': None}])
async def test_without_a_pit_it_falls_back_to_the_scroll_fetch(kwargs: dict[str, Any]) -> None:
    client = _Client(4, **kwargs)
    progress = _Progress()
    ids, matrix = await fetch_residual_embeddings_parallel(
        client, n_slices=3, extra_must=[{'range': {'rank': {'gte': 1}}}], progress=progress
    )
    assert ids == ['d0', 'd1', 'd2', 'd3']
    assert np.allclose(matrix, _unit([d[1] for d in client.docs]))
    assert client.bodies == []
    assert client.scroll_searches[0]['scroll'] == '2m'
    assert client.scroll_searches[0]['body']['size'] == 1000
    assert {'range': {'rank': {'gte': 1}}} in client.scroll_searches[0]['body']['query']['bool'][
        'filter'
    ]
    assert client.deleted == []


@pytest.mark.asyncio
async def test_a_failed_pit_delete_is_swallowed() -> None:
    client = _Client(2, fail_delete_pit=True)
    ids, _ = await fetch_residual_embeddings_parallel(client, n_slices=1)
    assert ids == ['d0', 'd1']


@pytest.mark.asyncio
async def test_an_empty_pool_returns_empty_ids_and_a_zero_matrix() -> None:
    client = _Client(0)
    ids, matrix = await fetch_residual_embeddings_parallel(client, n_slices=2)
    assert ids == []
    assert matrix.shape == (0, 0)
    assert matrix.dtype == np.float32


@pytest.mark.asyncio
async def test_the_scroll_fetch_uses_the_same_pool_gate_and_precount() -> None:
    kwargs: dict[str, Any] = {
        'include_candidate_clusters': False,
        'candidate_cluster_id_min': 100,
        'extra_must': [{'range': {'rank': {'gte': 1}}}],
    }
    pit_client = _Client(4)
    await fetch_residual_embeddings_parallel(pit_client, n_slices=1, **kwargs)
    scroll_client = _Client(4)
    progress = _Progress()
    ids, matrix = await residual_fetch.fetch_residual_embeddings(
        scroll_client, progress=progress, **kwargs
    )
    assert scroll_client.scroll_searches[0]['body']['query'] == pit_client.bodies[0]['query']
    assert scroll_client.scroll_searches[0]['body']['_source'] == [RESIDUAL_EMBEDDING_FIELD]
    assert scroll_client.counts == [
        {
            'index': scroll_client.counts[0]['index'],
            'body': {'query': pit_client.bodies[0]['query']},
        }
    ]
    assert ids == ['d0', 'd1', 'd2', 'd3']
    assert np.allclose(matrix, _unit([d[1] for d in scroll_client.docs]))
    assert progress.updates == [(0, 4), (4, 4)]
    assert progress.cancel_checks == 1
