"""Unit tests for cluster_id_normalize.py's run_update_by_query_polled.

The real bug this fixes: `client.update_by_query(..., wait_for_completion=True)`
on a large index (op_items, 347k+ docs) held the HTTP connection open
for the entire operation, then returned one huge response — which failed to
parse client-side ("Too many headers received") even though the operation
completed successfully server-side every time (confirmed live via
GET _tasks mid-run). opensearch-py's transport then silently retried the
*entire* multi-minute operation from scratch on every failure, hitting the
identical parse error each time — observed live: a real recluster job stuck
retrying the same update_by_query for 25+ minutes with zero net progress.

These tests exercise the fixed contract (submit async, poll the task,
surface the task's own error, respect the timeout) against a mocked
OpenSearch client — no live cluster needed for the unit-test tier.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.services.curation.clustering.id_normalize import run_update_by_query_polled


def _client(*, tasks_get_results: list[dict]) -> AsyncMock:
    client = AsyncMock()
    client.update_by_query = AsyncMock(return_value={'task': 'node1:42'})
    client.tasks = AsyncMock()
    client.tasks.get = AsyncMock(side_effect=tasks_get_results)
    return client


@pytest.mark.asyncio
async def test_submits_async_not_blocking():
    """The fix's whole point: never pass wait_for_completion=True."""
    client = _client(
        tasks_get_results=[
            {'completed': True, 'response': {'updated': 5, 'batches': 1}},
        ]
    )
    await run_update_by_query_polled(client, index='op_items', body={'query': {'match_all': {}}})
    _, kwargs = client.update_by_query.call_args
    assert kwargs['wait_for_completion'] is False


@pytest.mark.asyncio
async def test_returns_response_once_task_completes():
    client = _client(
        tasks_get_results=[
            {'completed': False},
            {'completed': False},
            {
                'completed': True,
                'response': {'updated': 219642, 'batches': 235, 'version_conflicts': 0},
            },
        ]
    )
    resp = await run_update_by_query_polled(
        client,
        index='op_items',
        body={'query': {'exists': {'field': 'class_id'}}},
        poll_interval_s=0.001,
    )
    assert resp == {'updated': 219642, 'batches': 235, 'version_conflicts': 0}
    assert client.tasks.get.call_count == 3
    assert client.tasks.get.call_args_list[0].kwargs['task_id'] == 'node1:42'


@pytest.mark.asyncio
async def test_raises_on_task_reported_error():
    """A real per-shard failure (not the header-parse artifact) must
    still surface as an error, not be silently swallowed."""
    client = _client(
        tasks_get_results=[
            {
                'completed': True,
                'error': {'type': 'search_phase_execution_exception', 'reason': 'boom'},
            },
        ]
    )
    with pytest.raises(RuntimeError, match='boom'):
        await run_update_by_query_polled(
            client, index='op_items', body={'query': {'match_all': {}}}
        )


@pytest.mark.asyncio
async def test_raises_timeout_error_when_never_completes():
    """A task that never reports completed=True must not poll forever —
    exercises the deadline path with a tiny timeout so the test is fast."""
    client = _client(tasks_get_results=[{'completed': False}] * 1000)
    with pytest.raises(TimeoutError):
        await run_update_by_query_polled(
            client,
            index='op_items',
            body={'query': {'match_all': {}}},
            poll_interval_s=0.001,
            timeout_s=0.01,
        )


@pytest.mark.asyncio
async def test_forwards_conflicts_and_refresh_params():
    client = _client(tasks_get_results=[{'completed': True, 'response': {}}])
    await run_update_by_query_polled(
        client,
        index='op_items',
        body={'query': {'match_all': {}}},
        conflicts='proceed',
        refresh=True,
    )
    _, kwargs = client.update_by_query.call_args
    assert kwargs['conflicts'] == 'proceed'
    assert kwargs['refresh'] is True


# =============================================================================
# force_cluster_id_equals_class_id must never pull an excluded item
# back into its class cluster.
# =============================================================================


@pytest.mark.asyncio
async def test_force_cluster_id_equals_class_id_excludes_excluded_items():
    from src.services.curation.clustering.id_normalize import force_cluster_id_equals_class_id

    client = AsyncMock()
    client.update_by_query = AsyncMock(return_value={'task': 'node1:1'})
    client.tasks = AsyncMock()
    client.tasks.get = AsyncMock(
        return_value={'completed': True, 'response': {'updated': 0, 'batches': 1}}
    )

    await force_cluster_id_equals_class_id(client)

    _, kwargs = client.update_by_query.call_args
    body = kwargs['body']
    assert {'term': {'class_excluded': True}} in body['query']['bool']['must_not']
    # Defense in depth: the script itself also no-ops on a freshly
    # excluded doc (a write racing the query match).
    assert 'class_excluded' in body['script']['source']
