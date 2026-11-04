"""The per-process order caches are bounded and expire by TTL (cachetools)."""

from __future__ import annotations

from cachetools import TTLCache

from src.services.curation import diverse_order
from src.services.curation.clustering import outliers


def test_order_caches_are_bounded_ttl_caches() -> None:
    assert isinstance(diverse_order._ORDER_CACHE, TTLCache)
    assert isinstance(outliers._CACHE, TTLCache)


def test_inserting_past_maxsize_evicts_the_oldest() -> None:
    for cache in (diverse_order._ORDER_CACHE, outliers._CACHE):
        cache.clear()
        for i in range(cache.maxsize + 1):
            cache[f'k{i}'] = {}

        assert len(cache) == cache.maxsize
        assert 'k0' not in cache
        cache.clear()
