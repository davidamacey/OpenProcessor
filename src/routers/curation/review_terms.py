"""Curation review router sub-module — raw-label aggregations that grow the registry.

``GET /review/unmatched_terms`` and ``GET /review/raw_label_clusters`` are
registered before the parameterised ``/review/{tab}`` (see
:mod:`src.routers.curation.review`) so the ``{tab}`` matcher does not eat them.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query

from src.routers.curation._common import OpenSearchDep, items_index, router
from src.services.curation.index_bootstrap import ensure_indexes
from src.services.curation.review_terms import (
    raw_label_clusters_body,
    raw_label_clusters_response,
    unmatched_terms_body,
    unmatched_terms_response,
)


@router.get('/review/unmatched_terms')
async def review_unmatched_terms(
    opensearch: OpenSearchDep,
    size: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """Aggregate the VLM's raw labels across every ``vlm_unmatched`` crop.

    Returns the top-N most common raw labels the VLM produced for crops the
    registry could not resolve. This is the main input to growing the
    registry: high-count labels are obvious candidates for new class
    entries (or new ``SYNONYMS`` mappings if the raw label is just a
    phrasing of an existing class).

    Once the registry (or the prompt pack's synonyms) has grown, run
    ``scripts/curation/reclassify_after_registry_growth.py`` with
    ``--label-prefix`` set to the prefix of the ``class_source`` /
    raw-label pair aggregated here; it promotes every ``<prefix>_unmatched``
    item whose raw label now resolves to ``<prefix>_reclassified`` (never
    setting ``class_validated``).

    Args:
        size: Maximum number of distinct raw labels to return. Capped at
            1000; default 100 (the long tail past 100 is usually noise).

    Returns:
        ``{"total_unmatched": <int>, "top_terms": [{"label": str, "count": int}, ...]}``
    """
    await ensure_indexes(opensearch)
    body = unmatched_terms_body(size)
    try:
        resp = await opensearch.search(index=items_index(), body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc

    return unmatched_terms_response(resp)


@router.get('/review/raw_label_clusters')
async def review_raw_label_clusters(
    opensearch: OpenSearchDep,
    size: int = Query(50, ge=1, le=500),
    samples_per_cluster: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    """Top-N hierarchical clusters of ``vlm_raw_label`` for the labeler UI.

    Surfaces the output of ``scripts/curation/cluster_raw_labels.py``
    (field contract: :mod:`src.services.curation.raw_label_clusters`) so
    the labeler can:

    1. Show fine-grained sub-classes the registry doesn't have yet
       (e.g. "ford f150" + "ford ranger" rolled up under
       ``ford_pickup`` → suggested registry parent ``pickup``).
    2. Surface candidate registry promotions ranked by crop volume.
    3. Let curators bulk-relabel a whole cluster at once instead of
       clicking through individual ``vlm_unmatched`` crops.

    The endpoint runs purely against the configured items index — no clustering
    happens here, only aggregation. Cluster ids / names are written by
    the offline script; if no crops have ``vlm_label_cluster_id`` yet
    the endpoint returns an empty list with a populated ``hint``.

    NOTE on route ordering: this endpoint MUST be declared before the
    parameterised ``/review/{tab}`` below — FastAPI dispatches the first
    matching route, and ``raw_label_clusters`` would otherwise be eaten
    by the ``{tab}`` matcher.

    Args:
        size: Number of clusters to return (default 50).
        samples_per_cluster: How many ``vlm_raw_label`` samples to
            include per cluster (default 5 — enough to read at a glance).

    Returns:
        ``{"status": "ok|empty", "clusters": [{cluster_id, cluster_name,
        n_crops, n_unmatched, sample_terms, parent_class_suggestion}, ...]}``
    """
    await ensure_indexes(opensearch)
    body = raw_label_clusters_body(size, samples_per_cluster)
    try:
        resp = await opensearch.search(index=items_index(), body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc

    return raw_label_clusters_response(resp)
