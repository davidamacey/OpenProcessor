"""The plumbing around the clustering methods: how embeddings arrive and assignments leave.

What is pinned here is the behaviour that the speed work changed (bulk requests in flight
together, a 429 retried instead of lost or fatal, the response decoder, the page-to-matrix
conversion) and what it must not change (the guarded script every document gets, each
document written once, a failure or a cancel stopping the pass). What the methods
compute is pinned separately by ``test_cluster_plumbing_golden.py``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from opensearchpy.exceptions import TransportError

from src.clients.opensearch.serializer import OrjsonSerializer
from src.services.curation.clustering import residual_fetch
from src.services.curation.clustering.cluster_write_guard import _guarded_class_cluster_write
from src.services.curation.clustering.orchestrator import assign_only_residuals, cluster_residuals
from src.services.curation.clustering.page_rows import hits_to_rows
from tests.curation._fake_items_pit import FakeItemsClient


N_DOCS = 4500  # more than two bulk chunks


@pytest.fixture
def ivf_state(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation.clustering.methods import ivf_store as ivf_store_mod

    class _Config:
        project_state_dir = tmp_path

    monkeypatch.setattr(ivf_store_mod, 'get_curation_config', _Config)


@pytest.fixture
def pool() -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(3)
    centers = rng.standard_normal((20, 8))
    vectors = centers[rng.integers(0, 20, N_DOCS)] + 0.2 * rng.standard_normal((N_DOCS, 8))
    return [f'd{i:05d}' for i in range(N_DOCS)], vectors.astype(np.float32)


class _Progress:
    """Records updates; cancels after ``cancel_after`` writes have been reported."""

    def __init__(self, cancel_after: int | None = None) -> None:
        self.cancel_after = cancel_after
        self.processed = 0

    def set_backend(self, **_: Any) -> None: ...

    def update(self, *, processed: int, total: int) -> None:  # noqa: ARG002
        self.processed = processed

    def raise_if_cancelled(self) -> None:
        if self.cancel_after is not None and self.processed >= self.cancel_after:
            raise RuntimeError('cancelled')


# ----------------------------------------------------------------- write-back


@pytest.mark.asyncio
async def test_chunks_are_written_concurrently_and_each_document_once(
    pool: tuple[list[str], np.ndarray], ivf_state: None
) -> None:
    client = FakeItemsClient(*pool)
    client.bulk_delay_s = 0.02
    bodies: list[list[dict[str, Any]]] = []
    real_bulk = client.bulk

    async def spy(*, body: list[dict[str, Any]], refresh: bool = False) -> dict[str, Any]:
        bodies.append(body)
        return await real_bulk(body=body, refresh=refresh)

    client.bulk = spy  # type: ignore[method-assign]
    summary = await cluster_residuals(client, clustering_method='ivf')
    assert summary['status'] == 'success'
    assert client.max_in_flight >= 2, 'bulk requests were sent one after another'
    ids = [b[i]['update']['_id'] for b in bodies for i in range(0, len(b), 2)]
    assert sorted(ids) == sorted(client.ids), 'every document exactly once'
    # Every action is the guarded update, unchanged.
    for body in bodies:
        for i in range(0, len(body), 2):
            params = body[i + 1]['script']['params']
            assert body[i + 1] == _guarded_class_cluster_write(params['cid'], params['dist'])


@pytest.mark.asyncio
async def test_concurrency_is_bounded(pool: tuple[list[str], np.ndarray], ivf_state: None) -> None:
    from src.services.curation.clustering.assignment_writer import BULK_CONCURRENCY

    client = FakeItemsClient(*pool)
    client.bulk_delay_s = 0.02
    await cluster_residuals(client, clustering_method='ivf')
    assert client.max_in_flight <= BULK_CONCURRENCY


@pytest.mark.asyncio
async def test_assign_only_writes_pages_concurrently(
    pool: tuple[list[str], np.ndarray], ivf_state: None
) -> None:
    client = FakeItemsClient(*pool)
    await cluster_residuals(client, clustering_method='ivf')
    full = dict(client.written)
    client.forget_assignments()
    client.bulk_delay_s = 0.02
    client.max_in_flight = 0
    summary = await assign_only_residuals(client, chunk_size=500)
    assert summary['n_assigned'] == N_DOCS
    assert client.max_in_flight >= 2
    assert set(client.written) == set(full)


@pytest.mark.asyncio
async def test_a_429_request_is_retried_not_dropped(
    pool: tuple[list[str], np.ndarray], ivf_state: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation.clustering import assignment_writer

    monkeypatch.setattr(assignment_writer, 'BULK_BACKOFF_S', 0.0)
    client = FakeItemsClient(*pool)
    real_bulk = client.bulk
    refused = {'n': 0}

    async def flaky(*, body: list[dict[str, Any]], refresh: bool = False) -> dict[str, Any]:
        if refused['n'] < 3:
            refused['n'] += 1
            raise TransportError(429, 'es_rejected_execution_exception', {})
        return await real_bulk(body=body, refresh=refresh)

    client.bulk = flaky  # type: ignore[method-assign]
    await cluster_residuals(client, clustering_method='ivf')
    assert refused['n'] == 3
    assert len(client.written) == N_DOCS


@pytest.mark.asyncio
async def test_items_rejected_with_429_are_retried_and_others_logged(
    pool: tuple[list[str], np.ndarray], ivf_state: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation.clustering import assignment_writer

    monkeypatch.setattr(assignment_writer, 'BULK_BACKOFF_S', 0.0)
    client = FakeItemsClient(*pool)
    real_bulk = client.bulk
    rejected_once: set[str] = set()
    seen_bad: list[dict[str, Any]] = []
    bad_id = client.ids[7]

    async def partial(*, body: list[dict[str, Any]], refresh: bool = False) -> dict[str, Any]:
        keep: list[dict[str, Any]] = []
        items: list[dict[str, Any]] = []
        for i in range(0, len(body), 2):
            doc_id = body[i]['update']['_id']
            if doc_id == bad_id:
                items.append({'update': {'_id': doc_id, 'status': 400, 'error': 'mapper'}})
            elif i % 3 == 0 and doc_id not in rejected_once:
                rejected_once.add(doc_id)
                items.append({'update': {'_id': doc_id, 'status': 429, 'error': 'rejected'}})
            else:
                keep += body[i : i + 2]
                items.append({'update': {'_id': doc_id, 'status': 200}})
        await real_bulk(body=keep, refresh=refresh)
        seen_bad.extend(it for it in items if it['update']['status'] != 200)
        return {'errors': True, 'items': items}

    client.bulk = partial  # type: ignore[method-assign]
    await cluster_residuals(client, clustering_method='ivf')
    assert rejected_once, 'the fake never rejected anything'
    assert set(client.written) == set(client.ids) - {bad_id}


@pytest.mark.asyncio
async def test_a_failure_other_than_429_still_stops_the_pass(
    pool: tuple[list[str], np.ndarray], ivf_state: None
) -> None:
    client = FakeItemsClient(*pool)

    async def broken(*, body: list[dict[str, Any]], refresh: bool = False) -> dict[str, Any]:
        raise TransportError(500, 'boom', {})

    client.bulk = broken  # type: ignore[method-assign]
    with pytest.raises(TransportError):
        await cluster_residuals(client, clustering_method='ivf')


@pytest.mark.asyncio
async def test_a_cancel_stops_the_writes(
    pool: tuple[list[str], np.ndarray], ivf_state: None
) -> None:
    client = FakeItemsClient(*pool)
    client.bulk_delay_s = 0.01
    progress = _Progress(cancel_after=1)
    with pytest.raises(RuntimeError, match='cancelled'):
        await cluster_residuals(client, clustering_method='ivf', progress=progress)
    calls_at_cancel = client.bulk_calls
    assert len(client.written) < N_DOCS
    import asyncio

    await asyncio.sleep(0.1)
    assert client.bulk_calls == calls_at_cancel, 'writes continued after the cancel'


# ----------------------------------------------------------------- fetch


def test_the_response_decoder_is_orjson_and_agrees_with_json() -> None:
    rng = np.random.default_rng(1)
    doc = {'hits': {'hits': [{'_id': 'a', '_source': {'v': rng.standard_normal(64).tolist()}}]}}
    text = json.dumps(doc)
    assert OrjsonSerializer().loads(text) == json.loads(text)


def test_the_response_decoder_does_not_use_stdlib_json_for_normal_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError('stdlib json.loads was used')

    monkeypatch.setattr('opensearchpy.serializer.json.loads', refuse)
    assert OrjsonSerializer().loads('{"a": [1.5, 2]}') == {'a': [1.5, 2]}


def test_the_response_decoder_falls_back_for_what_orjson_refuses() -> None:
    assert np.isnan(OrjsonSerializer().loads('{"a": NaN}')['a'])
    assert OrjsonSerializer().loads('{"a": Infinity}') == {'a': float('inf')}
    assert OrjsonSerializer().loads('{"a": "\\ud800"}') == json.loads('{"a": "\\ud800"}')


def test_hits_to_rows_matches_the_per_document_conversion() -> None:
    rng = np.random.default_rng(2)
    hits: list[dict[str, Any]] = [
        {'_id': f'h{i}', '_source': {'pe_embedding': rng.standard_normal(16).tolist()}}
        for i in range(9)
    ]
    hits.insert(4, {'_id': 'no-vector', '_source': {}})
    hits.insert(0, {'_id': 'no-source'})
    ids, matrix = hits_to_rows(hits, 'pe_embedding')
    kept = [h for h in hits if (h.get('_source') or {}).get('pe_embedding') is not None]
    assert ids == [h['_id'] for h in kept]
    want = np.vstack([np.asarray(h['_source']['pe_embedding'], dtype=np.float32) for h in kept])
    assert matrix is not None
    assert matrix.dtype == np.float32
    assert np.array_equal(matrix, want)


def test_hits_to_rows_of_a_page_without_vectors_is_empty() -> None:
    assert hits_to_rows([{'_id': 'x', '_source': {}}], 'pe_embedding') == (
        [],
        None,
    )


@pytest.mark.asyncio
async def test_parallel_fetch_returns_the_pool_in_document_id_order(
    pool: tuple[list[str], np.ndarray],
) -> None:
    """The rows come back sorted by document id: some methods sample by position (#372)."""
    client = FakeItemsClient(*pool)
    ids, matrix = await residual_fetch.fetch_residual_embeddings_parallel(
        client, n_slices=4, page_size=700
    )
    expected = sorted(range(N_DOCS), key=client.ids.__getitem__)
    assert ids == [client.ids[p] for p in expected]
    unit = pool[1][expected]
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    assert np.array_equal(matrix, unit.astype(np.float32)) or np.allclose(matrix, unit, atol=1e-7)


@pytest.mark.asyncio
async def test_assign_only_reads_through_point_in_time_slices(
    pool: tuple[list[str], np.ndarray], ivf_state: None
) -> None:
    client = FakeItemsClient(*pool)
    await cluster_residuals(client, clustering_method='ivf')
    full = dict(client.written)
    client.forget_assignments()
    summary = await assign_only_residuals(client, chunk_size=500)
    assert client.pit_searches >= 2, 'the pool was not read through point-in-time slices'
    assert summary['n_assigned'] == N_DOCS
    assert client.written == full, 'a different order of pages changed an assignment'


@pytest.mark.asyncio
async def test_assign_only_skips_what_is_already_stored_exactly(
    pool: tuple[list[str], np.ndarray], ivf_state: None
) -> None:
    client = FakeItemsClient(*pool)
    await cluster_residuals(client, clustering_method='ivf')
    stored = dict(client.written)
    client.written.clear()
    again = await assign_only_residuals(client)
    assert again['n_assigned'] == N_DOCS
    assert again['n_unchanged'] == N_DOCS
    assert client.written == {}, 'unchanged documents were written again'
    # A stale sub-cluster id is not "already stored": the update removes it.
    client.docs[3]['cluster_subid'] = '5a'
    third = await assign_only_residuals(client)
    assert third['n_unchanged'] == N_DOCS - 1
    assert set(client.written) == {client.ids[3]}
    assert client.written[client.ids[3]] == stored[client.ids[3]]
    assert 'cluster_subid' not in client.docs[3]
