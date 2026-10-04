"""#123: the snapshot-gauge refresh loop runs outside any request, so it must bind
each project itself. These tests run it against a real ``AsyncOpenSearch`` behind the
real project guard (only the network transport is faked)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from _fake_project_registry import StaticProjectRegistry
from opensearchpy import AsyncOpenSearch
from prometheus_client import REGISTRY

from src.config.curation import IndexRole, base_curation_config
from src.config.projects import ProjectRecord, resources_for_new
from src.services.curation import ops_metrics_refresh as refresh
from src.services.projects.guard import install_project_guard


pytestmark = pytest.mark.unbound


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


class _Bottom:
    """The network edge: answers searches and ``_cat/indices``; records every request."""

    def __init__(self, docs: dict[str, int]) -> None:
        self.docs = docs
        self.seen: list[tuple[str, str]] = []

    async def perform_request(self, method: str, url: str, params=None, body=None, **_kw):  # noqa: ARG002
        self.seen.append((method, url))
        if url.startswith('/_cat/indices/'):
            names = url.removeprefix('/_cat/indices/').split(',')
            return [
                {'index': n, 'pri': '1', 'rep': '1', 'store.size': str(1000 + len(n))}
                for n in names
            ]
        index = url.split('/')[1]
        n = self.docs[index]
        return {
            'aggregations': {
                'segment': {'doc_count': n, 'oldest': {'value': 1_000_000.0}},
                'label': {'doc_count': 1, 'oldest': {'value': None}},
                'embed': {'doc_count': 0, 'oldest': {'value': None}},
                'embedded': {'doc_count': n * 10},
                'failed': {'doc_count': 0},
            }
        }

    async def close(self) -> None:
        return None


@pytest.fixture
def stack(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('src.services.detection.segmenter_http.first_segmenter_url', lambda: None)
    alpha, beta = _record('gauge-alpha'), _record('gauge-beta')
    registry = StaticProjectRegistry([alpha, beta])
    docs = {
        alpha.resources.indexes[IndexRole.ITEMS]: 3,
        beta.resources.indexes[IndexRole.ITEMS]: 8,
    }
    bottom = _Bottom(docs)
    client = AsyncOpenSearch(hosts=['http://127.0.0.1:9'])
    client.transport = bottom  # type: ignore[assignment]
    install_project_guard(client, registry)
    return client, registry, bottom, alpha, beta


def _sample(name: str, **labels: str) -> float | None:
    return REGISTRY.get_sample_value(name, labels)


@pytest.mark.asyncio
async def test_refresh_populates_every_project_through_the_real_guard(stack) -> None:
    client, registry, bottom, alpha, beta = stack
    await refresh.refresh_once(client, registry.active_projects())

    assert _sample('op_queue_depth', queue='segment', project='gauge-alpha') == 3
    assert _sample('op_queue_depth', queue='segment', project='gauge-beta') == 8
    assert _sample('op_embedding_state_items', project='gauge-alpha', state='embedded') == 30
    assert _sample('op_embedding_state_items', project='gauge-beta', state='embedded') == 80
    assert _sample('op_opensearch_shards', project='gauge-alpha', index_role='items') == 2
    assert _sample('op_opensearch_shards', project='gauge-beta', index_role='items') == 2
    assert (_sample('op_opensearch_store_bytes', project='gauge-beta', index_role='items') or 0) > 0

    owned = {n for r in (alpha, beta) for n in r.resources.indexes.values()}
    for _method, url in bottom.seen:
        if url.startswith('/_cat/indices/'):
            assert owned.issuperset(url.removeprefix('/_cat/indices/').split(','))
            assert '*' not in url
        else:
            assert url.split('/')[1] in owned


@pytest.mark.asyncio
async def test_gauges_appear_in_a_scrape(stack) -> None:
    from src.core.metrics import render_metrics

    client, registry, _bottom, _a, _b = stack
    await refresh.refresh_once(client, registry.active_projects())
    body = render_metrics()[0].decode()
    for line in (
        'op_queue_depth{project="gauge-beta",queue="segment"} 8.0',
        'op_embedding_state_items{project="gauge-alpha",state="embedded"} 30.0',
    ):
        assert line in body
    assert 'op_opensearch_store_bytes{index_role="items",project="gauge-alpha"}' in body
    assert 'op_queue_oldest_item_age_seconds{queue="segment"}' in body


@pytest.mark.asyncio
async def test_one_failing_project_keeps_its_previous_value(stack) -> None:
    client, registry, bottom, _alpha, beta = stack
    await refresh.refresh_once(client, registry.active_projects())
    assert _sample('op_queue_depth', queue='segment', project='gauge-beta') == 8
    bottom.docs[beta.resources.indexes[IndexRole.ITEMS]] = 99

    original = bottom.perform_request

    async def _flaky(method: str, url: str, **kw: Any):
        if beta.resources.indexes[IndexRole.ITEMS] in url:
            raise ConnectionError('down')
        return await original(method, url, **kw)

    bottom.perform_request = _flaky  # type: ignore[method-assign]
    await refresh.refresh_once(client, registry.active_projects())
    assert _sample('op_queue_depth', queue='segment', project='gauge-alpha') == 3
    assert _sample('op_queue_depth', queue='segment', project='gauge-beta') == 8
    json.dumps(bottom.docs)
