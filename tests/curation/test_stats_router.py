"""Unit tests for the /curation/stats/dataset class_source rollup.

Guards against a crop with no class_source (e.g. freshly unlabel_crop'd)
dropping out of the labeled.* buckets entirely instead of landing in
'other', which would break the "buckets sum to total_crops" invariant
``tests/integration/test_stats_router.py`` checks against a fuller
fixture.
"""

from __future__ import annotations

from src.routers.curation.stats import _build_dataset_query_body, _rollup_class_sources


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
