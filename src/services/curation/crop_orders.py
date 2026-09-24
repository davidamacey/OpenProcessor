"""Computed member orders for ``GET /crops`` (``order=outliers|core_first|diverse``).

Split out of the crops router (LOC ceiling). Each order ranks the pool
the request's filters matched, pages the ranked ids, and hydrates them
through the caller's ``fetch_items``. ``None`` means "fall back to the
request's plain ``sort``" (pool too large, no embeddings, feature off).

``core_first`` (DQ-M3) is the cluster view's cut-line order: members
nearest their cluster's centroid first. The centroid is computed live
from the matched members (the same member-mean centroid as
``order=outliers``), and each served item's ``cluster_distance`` /
``cluster_similarity`` / ``cluster_is_core`` is overwritten with that live
value, so the cut line (first non-core item) and the order always agree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.cluster_ids import CORE_SIMILARITY_MIN, cluster_similarity
from src.services.curation.crop_browse import crops_page, embedding_pool_query_and_count


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


CLUSTER_ORDERS = ('outliers', 'core_first')


def with_live_distance(item: dict[str, Any], distance: float | None) -> dict[str, Any]:
    """Overlay a live centroid distance on a served item."""
    similarity = cluster_similarity(distance)
    item['cluster_distance'] = distance
    item['cluster_similarity'] = similarity
    item['cluster_is_core'] = None if similarity is None else similarity >= CORE_SIMILARITY_MIN
    return item


def _page(ids: list[str], page: int, page_size: int) -> list[str]:
    start = (page - 1) * page_size
    return ids[start : start + page_size]


async def ordered_crops_page(
    opensearch: Any,
    *,
    index: str,
    order: str,
    query_clause: dict[str, Any],
    cluster_id: int | None,
    page: int,
    page_size: int,
    k: int | None,
    n_pool: int,
    fetch_items: Callable[[Any, list[str]], Awaitable[list[dict[str, Any]]]],
) -> dict[str, Any] | None:
    """The ``crops_page`` envelope for a computed order, or ``None``."""
    if order in CLUSTER_ORDERS and cluster_id is not None:
        from src.services.curation.clustering.outliers import (
            OUTLIER_EMBEDDING_FIELD,
            compute_centroid_distances,
        )

        # F-16: pool query + exact count scoped to the embedding-bearing subset.
        pool_query, pool_count = await embedding_pool_query_and_count(
            opensearch, index, query_clause, OUTLIER_EMBEDDING_FIELD
        )
        distances = await compute_centroid_distances(
            opensearch, index, pool_query, current_count=pool_count
        )
        if distances is None:
            return None
        if order == 'outliers':
            ranked = sorted(distances, key=lambda i: (-distances[i], i))
        else:
            ranked = sorted(distances, key=lambda i: (distances[i], i))
        items = await fetch_items(opensearch, _page(ranked, page, page_size))
        if order == 'core_first':
            items = [with_live_distance(it, distances.get(it['crop_id'])) for it in items]
        return crops_page(
            total=len(ranked),
            page=page,
            page_size=page_size,
            crops=items,
            method=order,
            n_pool=n_pool,
        )

    if order == 'diverse':
        from src.routers.curation.select import EMBEDDING_FIELD, compute_diverse_order

        pool_query, pool_count = await embedding_pool_query_and_count(
            opensearch, index, query_clause, EMBEDDING_FIELD
        )
        diverse_ids = await compute_diverse_order(
            opensearch, index, pool_query, current_count=pool_count, k=k
        )
        if diverse_ids is None:
            return None
        items = await fetch_items(opensearch, _page(diverse_ids, page, page_size))
        return crops_page(
            total=len(diverse_ids),
            page=page,
            page_size=page_size,
            crops=items,
            method='diverse',
            n_pool=n_pool,
        )
    return None


__all__ = ['CLUSTER_ORDERS', 'ordered_crops_page', 'with_live_distance']
