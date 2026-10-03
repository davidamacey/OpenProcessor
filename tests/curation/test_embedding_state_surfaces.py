"""The places a client learns that some items have no vector."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.config.region_fields import get_region_fields
from src.services.curation import semantic_search
from src.services.curation.crop_orders import ordered_crops_page
from src.services.curation.item_filter import ItemFilter
from src.services.curation.review_empty_reason import compute_empty_reason
from src.services.curation.stats_embedding import embedding_aggregations, embedding_summary
from src.services.projects.combine.copy_docs import transform_item


def test_stats_summary_splits_reasons_and_counts_unknown_legacy() -> None:
    aggs = {
        'embedded_items': {'doc_count': 6},
        'embedding_states': {
            'buckets': [
                {'key': 'embedded', 'doc_count': 5},
                {'key': 'failed', 'doc_count': 2},
                {'key': '__none__', 'doc_count': 3},
            ]
        },
    }
    assert embedding_summary(aggs, 10) == {
        'embedded': 6,
        'not_embedded': 4,
        'by_state': {
            'embedded': 5,
            'not_selected': 0,
            'deferred': 0,
            'failed': 2,
            'unknown': 3,
        },
    }
    assert set(embedding_aggregations()) == {
        'embedding_states',
        'embedded_items',
        'legacy_embedded_items',
    }


@pytest.mark.asyncio
async def test_semantic_search_reports_items_it_could_not_search() -> None:
    os_fake = AsyncMock()
    os_fake.search = AsyncMock(return_value={'hits': {'total': {'value': 0}, 'hits': []}})
    os_fake.count = AsyncMock(return_value={'count': 4})
    encoder = AsyncMock()
    encoder.encode_text = lambda texts: np.zeros((len(texts), 3), dtype=np.float32)

    out = await semantic_search.semantic_text_search(
        opensearch=os_fake,
        pe_encoder=encoder,
        executor=None,
        query='dog',
        page=1,
        page_size=10,
    )

    assert out['unembedded_in_scope'] == 4
    body = os_fake.count.call_args.kwargs['body']['query']['bool']['filter']
    assert {'bool': {'must_not': [{'exists': {'field': 'pe_embedding'}}]}} in body


@pytest.mark.asyncio
async def test_ordered_page_reports_how_many_it_could_not_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _diverse(*_: Any, **__: Any) -> list[str]:
        return ['a', 'b']

    monkeypatch.setattr('src.routers.curation.select.compute_diverse_order', _diverse)
    os_fake = AsyncMock()
    os_fake.count = AsyncMock(return_value={'count': 7})

    async def _fetch(_: Any, ids: list[str]) -> list[dict[str, Any]]:
        return [{'crop_id': i} for i in ids]

    page = await ordered_crops_page(
        os_fake,
        index='items',
        order='diverse',
        query_clause={'match_all': {}},
        cluster_id=None,
        item_filter=ItemFilter(),
        page=1,
        page_size=10,
        k=None,
        n_pool=10,
        fetch_items=_fetch,
    )
    assert page is not None
    assert (page['n_pool'], page['n_unembedded']) == (10, 3)
    # The ordering cannot rank them and never embeds: it hands back the request that would.
    request = page['suggested_reprocess']
    assert request['scopes'] == ['embed']
    assert request['dry_run'] is True
    assert request['targets']['filter']['embedding_state'] == ['not_selected', 'deferred', 'failed']


@pytest.mark.asyncio
async def test_empty_all_queue_says_items_are_not_embedded() -> None:
    os_fake = AsyncMock()
    os_fake.count = AsyncMock(return_value={'count': 5})
    reason = await compute_empty_reason('all', None, os_fake)
    assert '5 items have no embedding' in reason

    os_fake.count = AsyncMock(return_value={'count': 0})
    assert await compute_empty_reason('all', None, os_fake) == 'no items match'


def test_combine_marks_a_dropped_vector_deferred() -> None:
    item = {
        'crop_id': 'c',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'pe_embedding': [0.1, 0.2],
        'embedding_state': 'embedded',
    }
    doc, dropped = transform_item(
        item,
        target_image_id='i',
        target_image_path_='/p',
        target_class=None,
        job_id='j',
        origin_project='o',
        now='n',
        embedding_dim=3,
        fields=get_region_fields(),
    )
    assert dropped is True
    assert 'pe_embedding' not in doc
    assert doc['embedding_state'] == 'deferred'

    kept, dropped = transform_item(
        item,
        target_image_id='i',
        target_image_path_='/p',
        target_class=None,
        job_id='j',
        origin_project='o',
        now='n',
        embedding_dim=2,
        fields=get_region_fields(),
    )
    assert dropped is False
    assert kept['embedding_state'] == 'embedded'


@pytest.mark.asyncio
async def test_pipeline_snapshot_counts_items_waiting_for_a_vector() -> None:
    from src.routers.curation.pipeline_health import pipeline_health_snapshot

    os_fake = AsyncMock()
    os_fake.search = AsyncMock(
        return_value={
            'hits': {'total': {'value': 9}},
            'aggregations': {'unembedded': {'doc_count': 4}},
        }
    )
    snap = await pipeline_health_snapshot(os_fake)
    assert snap['unembedded'] == 4
    aggs = os_fake.search.call_args.kwargs['body']['aggs']
    assert aggs['unembedded']['filter'] == {
        'bool': {'must_not': [{'exists': {'field': 'pe_embedding'}}]}
    }


def test_legacy_items_with_a_vector_count_as_embedded_in_by_state() -> None:
    aggs = {
        'embedded_items': {'doc_count': 2740},
        'legacy_embedded_items': {'doc_count': 2740},
        'embedding_states': {'buckets': [{'key': '__none__', 'doc_count': 2740}]},
    }
    summary = embedding_summary(aggs, 2740)
    assert summary['embedded'] == summary['by_state']['embedded'] == 2740
    assert summary['by_state']['unknown'] == 0
    assert sum(summary['by_state'].values()) == 2740
