"""Unit tests for the /curation/stats/dataset class_source rollup.

Guards against a crop with no class_source (e.g. freshly unlabel_crop'd)
dropping out of the labeled.* buckets entirely instead of landing in
'other', which would break the "buckets sum to total_crops" invariant
``tests/integration/test_stats_router.py`` checks against a fuller
fixture.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.routers.curation.stats import (
    _build_dataset_query_body,
    _rollup_class_sources,
    stats_dataset,
)


def test_missing_class_source_bucket_key_lands_in_other() -> None:
    """A doc with no class_source.keyword value must not be dropped.

    The terms aggregation is configured with missing='__none__' (see
    ``_build_dataset_query_body``'s 'class_sources' agg body) precisely
    so a doc with no class_source still gets a bucket. This asserts the
    rollup function correctly routes that sentinel key to 'other' rather
    than matching it against a real-provenance prefix by accident.
    """
    buckets = [
        {'key': 'human', 'doc_count': 5},
        {'key': '__none__', 'doc_count': 3},
    ]
    rollup = _rollup_class_sources(buckets)
    assert rollup['by_human'] == 5
    assert rollup['other'] == 3
    assert sum(rollup.values()) == 8


def test_class_sources_agg_configures_a_missing_bucket() -> None:
    """Regression guard: the terms agg must request a `missing` bucket.

    Without it, OpenSearch's terms agg silently excludes docs with no
    value for the field, so labeled.* buckets undercount total_crops by
    exactly the number of crops with no class_source (real example: a
    crop right after DELETE /curation/crops/{id}/label, before any
    relabel).
    """
    import inspect

    from src.config.region_fields import get_region_fields

    src = inspect.getsource(_build_dataset_query_body)
    assert "'missing': '__none__'" in src, (
        "the 'class_sources' terms aggregation must set missing='__none__' "
        'so docs with no class_source.keyword value still get a bucket'
    )
    # Also prove the body actually built from the source carries it —
    # not just present in a comment.
    body = _build_dataset_query_body(get_region_fields())
    assert body['aggs']['class_sources']['terms']['missing'] == '__none__'


def test_dataset_query_body_scopes_class_source_rollup_to_docs_with_class_id() -> None:
    """D1 regression guard.

    The query body must carry sibling aggs that split the class_source
    breakdown by whether the doc has a class_id, so the rollup used for
    ``labeled.*`` can exclude class-less VLM proposals.
    """
    from src.config.region_fields import get_region_fields

    body = _build_dataset_query_body(get_region_fields())
    assert body['aggs']['class_sources_with_class']['filter'] == {'exists': {'field': 'class_id'}}
    assert (
        body['aggs']['class_sources_with_class']['aggs']['by_source']['terms']['field']
        == 'class_source'
    )
    assert body['aggs']['class_sources_no_class']['filter'] == {
        'bool': {'must_not': [{'exists': {'field': 'class_id'}}]}
    }


@pytest.mark.asyncio
async def test_stats_dataset_vlm_labeled_excludes_class_less_vlm_proposals() -> None:
    """D1: a crop the VLM answered but never matched to a registry class
    (class_source='vlm_unmatched', class_id=None) must not be counted
    under labeled.by_vlm -- it belongs under unlabeled.vlm_no_class."""
    os_client = AsyncMock()
    os_client.search = AsyncMock(
        return_value={
            'hits': {'total': {'value': 3}},
            'aggregations': {
                'class_sources_with_class': {
                    'by_source': {'buckets': [{'key': 'vlm', 'doc_count': 1}]},
                },
                'class_sources_no_class': {
                    'by_source': {'buckets': [{'key': 'vlm_unmatched', 'doc_count': 2}]},
                },
            },
        }
    )
    body = await stats_dataset(os_client)
    assert body['labeled']['by_vlm'] == 1
    assert body['unlabeled']['vlm_no_class'] == 2


@pytest.mark.asyncio
async def test_stats_dataset_uses_request_cache() -> None:
    """request_cache=True lets identical size:0 stats queries
    within an OpenSearch shard-cache refresh window skip re-execution."""
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value={'hits': {'total': {'value': 0}}, 'aggregations': {}})
    await stats_dataset(os_client)
    assert os_client.search.await_args is not None
    assert os_client.search.await_args.kwargs['request_cache'] is True
