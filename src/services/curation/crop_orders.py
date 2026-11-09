"""Computed member orders for ``GET /crops`` (``order=outliers|core_first|diverse``).

Split out of the crops router (LOC ceiling). Each order ranks the pool
the request's filters matched, pages the ranked ids, and hydrates them
through the caller's ``fetch_items``. ``None`` means "fall back to the
request's plain ``sort``" (pool too large, no embeddings, feature off).

``core_first`` is the cluster view's cut-line order: members the partition scored best against
their cluster first (a native ``cluster_score`` sort of the cluster's members, one page of
documents). ``outliers`` computes the member-mean centroid live from the matched members.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.crop_browse import crops_page, embedding_pool_query_and_count
from src.services.curation.detections_summary import suggested_embed_request
from src.services.embedding_spaces.readers import item_vector_field


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.services.curation.item_filter import ItemFilter


CLUSTER_ORDERS = ('outliers', 'core_first')


def _embed_suggestion(item_filter: ItemFilter, n_unembedded: int) -> dict[str, Any] | None:
    """The embed request for the items an ordering could not rank (none when
    every item in the pool has a vector). Reads never embed: the client POSTs it."""
    if n_unembedded <= 0:
        return None
    return suggested_embed_request(item_filter).model_dump(mode='json')


def _page(ids: list[str], page: int, page_size: int) -> list[str]:
    start = (page - 1) * page_size
    return ids[start : start + page_size]


async def ordered_crops_page(
    opensearch: Any,
    *,
    index: str,
    order: str,
    query_clause: dict[str, Any],
    cluster_uid: str | None,
    item_filter: ItemFilter,
    page: int,
    page_size: int,
    k: int | None,
    n_pool: int,
    fetch_items: Callable[[Any, list[str]], Awaitable[list[dict[str, Any]]]],
) -> dict[str, Any] | None:
    """The ``crops_page`` envelope for a computed order, or ``None``."""
    if order in CLUSTER_ORDERS and cluster_uid is not None:
        from src.services.curation.clustering.outliers import compute_centroid_distances

        # Pool query + exact count scoped to the embedding-bearing subset.
        vector_field = await item_vector_field()
        pool_query, pool_count = await embedding_pool_query_and_count(
            opensearch, index, query_clause, vector_field
        )
        if order == 'core_first':
            resp = await opensearch.search(
                index=index,
                body={
                    'query': pool_query,
                    'from': (page - 1) * page_size,
                    'size': page_size,
                    '_source': False,
                    'sort': [
                        {'cluster_score': {'order': 'desc', 'missing': '_last'}},
                        {'crop_id': {'order': 'asc'}},
                    ],
                },
            )
            ids = [h['_id'] for h in (resp.get('hits') or {}).get('hits') or []]
            return crops_page(
                total=pool_count,
                page=page,
                page_size=page_size,
                crops=await fetch_items(opensearch, ids),
                method=order,
                n_pool=n_pool,
                n_unembedded=max(0, n_pool - pool_count),
                suggested_reprocess=_embed_suggestion(item_filter, n_pool - pool_count),
            )
        distances = await compute_centroid_distances(
            opensearch, index, pool_query, embedding_field=vector_field, current_count=pool_count
        )
        if distances is None:
            return None
        ranked = sorted(distances, key=lambda i: (-distances[i], i))
        items = await fetch_items(opensearch, _page(ranked, page, page_size))
        return crops_page(
            total=len(ranked),
            page=page,
            page_size=page_size,
            crops=items,
            method=order,
            n_pool=n_pool,
            n_unembedded=max(0, n_pool - pool_count),
            suggested_reprocess=_embed_suggestion(item_filter, n_pool - pool_count),
        )

    if order == 'diverse':
        from src.services.curation.diverse_order import compute_diverse_order

        vector_field = await item_vector_field()
        pool_query, pool_count = await embedding_pool_query_and_count(
            opensearch, index, query_clause, vector_field
        )
        diverse_ids = await compute_diverse_order(
            opensearch,
            index,
            pool_query,
            embedding_field=vector_field,
            current_count=pool_count,
            k=k,
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
            n_unembedded=max(0, n_pool - pool_count),
            suggested_reprocess=_embed_suggestion(item_filter, n_pool - pool_count),
        )
    return None


__all__ = ['CLUSTER_ORDERS', 'ordered_crops_page']
