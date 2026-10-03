"""P1: read-only shard/heap capacity check (§2.3). P1 only serves this;
P3 adds create-time enforcement."""

from __future__ import annotations

import asyncio

import pytest

from src.services.projects import capacity as capacity_mod
from src.services.projects.capacity import capacity_status, per_project_shards


class _FakeTransport:
    def __init__(self, health, settings, jvm_stats) -> None:
        self._responses = {
            '/_cluster/health': health,
            '/_cluster/settings': settings,
            '/_nodes/stats/jvm': jvm_stats,
        }

    async def perform_request(self, method, url, params=None, **kwargs):  # noqa: ARG002
        return self._responses[url]


class _FakeClient:
    def __init__(self, transport) -> None:
        self.transport = transport


def _client(active_shards: int, max_shards_per_node: int, heap_bytes: int) -> _FakeClient:
    health = {'active_shards': active_shards, 'number_of_data_nodes': 1}
    settings = {'persistent': {'cluster.max_shards_per_node': max_shards_per_node}, 'transient': {}}
    jvm = {'nodes': {'n1': {'jvm': {'mem': {'heap_max_in_bytes': heap_bytes}}}}}
    return _FakeClient(_FakeTransport(health, settings, jvm))


@pytest.fixture(autouse=True)
def _clear_cache():
    capacity_mod._cache = None
    yield
    capacity_mod._cache = None


def test_per_project_shards_matches_resources_for_new() -> None:
    from src.config.curation import base_curation_config
    from src.config.projects import resources_for_new

    expected = len(set(resources_for_new('x', base_curation_config()).indexes.values()))
    assert per_project_shards() == expected


def test_status_ok_with_plenty_of_headroom() -> None:
    client = _client(active_shards=10, max_shards_per_node=1000, heap_bytes=8 * 1024**3)
    result = asyncio.run(capacity_status(client))
    assert result is not None
    assert result.status == 'ok'


def test_status_warn_near_soft_limit() -> None:
    # 2 GB heap -> soft_limit ~40; put active_shards near it.
    client = _client(active_shards=39, max_shards_per_node=1000, heap_bytes=2 * 1024**3)
    result = asyncio.run(capacity_status(client, extra_shards=7))
    assert result is not None
    assert result.status == 'warn'


def test_status_blocked_past_hard_limit() -> None:
    client = _client(active_shards=995, max_shards_per_node=1000, heap_bytes=2 * 1024**3)
    result = asyncio.run(capacity_status(client, extra_shards=7))
    assert result is not None
    assert result.status == 'blocked'


def test_unreachable_cluster_returns_none() -> None:
    class _BoomTransport:
        async def perform_request(self, *args, **kwargs):  # noqa: ARG002
            raise ConnectionError('unreachable')

    client = _FakeClient(_BoomTransport())
    result = asyncio.run(capacity_status(client))
    assert result is None


def test_result_is_cached_for_ttl() -> None:
    calls = []

    class _CountingTransport(_FakeTransport):
        async def perform_request(self, method, url, params=None, **kwargs):
            calls.append(url)
            return await super().perform_request(method, url, params=params, **kwargs)

    client = _client(active_shards=1, max_shards_per_node=1000, heap_bytes=2 * 1024**3)
    client.transport = _CountingTransport(
        {'active_shards': 1, 'number_of_data_nodes': 1},
        {'persistent': {'cluster.max_shards_per_node': 1000}, 'transient': {}},
        {'nodes': {'n1': {'jvm': {'mem': {'heap_max_in_bytes': 2 * 1024**3}}}}},
    )
    asyncio.run(capacity_status(client))
    first_call_count = len(calls)
    asyncio.run(capacity_status(client))
    assert len(calls) == first_call_count  # cache hit, no new transport calls


def test_heap_sum_excludes_non_data_nodes() -> None:
    """m8: a master/coordinating-only node's heap must not inflate the
    shard-serving capacity this cluster actually has."""
    from src.services.projects.capacity import invalidate_capacity_cache

    health = {'active_shards': 1, 'number_of_data_nodes': 1}
    settings = {'persistent': {'cluster.max_shards_per_node': 1000}, 'transient': {}}
    jvm = {
        'nodes': {
            'data1': {'roles': ['data'], 'jvm': {'mem': {'heap_max_in_bytes': 2 * 1024**3}}},
            'master1': {'roles': ['master'], 'jvm': {'mem': {'heap_max_in_bytes': 6 * 1024**3}}},
        }
    }
    client = _FakeClient(_FakeTransport(health, settings, jvm))
    invalidate_capacity_cache()
    result = asyncio.run(capacity_status(client))
    assert result is not None
    assert result.heap_max_bytes == 2 * 1024**3, 'the master-only node must not count'


def test_warn_message_rounds_a_half_gb_heap_honestly() -> None:
    """m8: a real 0.5 GB heap must not format as "0 GB"."""
    client = _client(active_shards=9, max_shards_per_node=1000, heap_bytes=int(0.5 * 1024**3))
    result = asyncio.run(capacity_status(client, extra_shards=7))
    assert result is not None
    assert result.status == 'warn'
    assert '0.5 GB' in result.message, result.message


def test_invalidate_capacity_cache_forces_a_fresh_read() -> None:
    """m7: create/delete must bust the 10s cache immediately, not leave a
    concurrent caller reading a stale active_shards for up to 10s after
    a write that already landed."""
    from src.services.projects.capacity import invalidate_capacity_cache

    calls: list[str] = []

    class _CountingTransport(_FakeTransport):
        async def perform_request(self, method, url, params=None, **kwargs):
            calls.append(url)
            return await super().perform_request(method, url, params=params, **kwargs)

    client = _client(active_shards=1, max_shards_per_node=1000, heap_bytes=2 * 1024**3)
    client.transport = _CountingTransport(
        {'active_shards': 1, 'number_of_data_nodes': 1},
        {'persistent': {'cluster.max_shards_per_node': 1000}, 'transient': {}},
        {'nodes': {'n1': {'jvm': {'mem': {'heap_max_in_bytes': 2 * 1024**3}}}}},
    )
    asyncio.run(capacity_status(client))
    first_call_count = len(calls)
    invalidate_capacity_cache()
    asyncio.run(capacity_status(client))
    assert len(calls) > first_call_count, 'invalidation must force a real re-read, not a cache hit'


def test_wire_names_the_binding_limit_and_the_total_after_a_create() -> None:
    heap_bound = asyncio.run(
        capacity_status(
            _client(active_shards=36, max_shards_per_node=1000, heap_bytes=2 * 1024**3),
            extra_shards=6,
        )
    )
    assert heap_bound is not None
    wire = heap_bound.to_wire()
    assert (wire['soft_limit'], wire['limit_source']) == (40, 'heap')
    assert wire['shards_after_create'] == 42
    assert wire['status'] == 'warn'
    assert 'every index in the cluster' in wire['message']

    capacity_mod._cache = None
    cluster_bound = asyncio.run(
        capacity_status(
            _client(active_shards=10, max_shards_per_node=30, heap_bytes=8 * 1024**3),
            extra_shards=6,
        )
    )
    assert cluster_bound is not None
    assert cluster_bound.to_wire()['limit_source'] == 'cluster_max_shards_per_node'
