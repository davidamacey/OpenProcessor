"""Computed member orders for ``GET /crops`` (``order=outliers|core_first|diverse``).

Split out of the crops router (LOC ceiling). Each order ranks the pool
the request's filters matched, pages the ranked ids, and hydrates them
through the caller's ``fetch_items``. ``None`` means "fall back to the
request's plain ``sort``" (pool too large, no embeddings, feature off).

``core_first`` is the cluster view's cut-line order: members
nearest their cluster's centroid first. Fast path: when every embedded
member carries a ``cluster_distance`` measured against this cluster
(written by the clustering geometry pass), the page is a native
``cluster_distance`` sort and each item's served ``cluster_is_core`` comes
from that same stored value, so order and cut line agree. Otherwise
(legacy / not yet measured / members moved in) the centroid is computed
live from the matched members (the same member-mean centroid as
``order=outliers``), each served item is overwritten with the live value,
and the distances are written back lazily so the next request is fast.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.curation.cluster_ids import CORE_SIMILARITY_MIN, cluster_similarity
from src.services.curation.crop_browse import crops_page, embedding_pool_query_and_count
from src.services.curation.detections_summary import suggested_embed_request


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.services.curation.item_filter import ItemFilter


logger = get_logger(__name__)

CLUSTER_ORDERS = ('outliers', 'core_first')

# (index, cluster_id) backfills in flight in this process, and strong refs to
# their tasks (the loop only keeps weak ones).
_BACKFILLING: set[tuple[str, int]] = set()
_TASKS: set[asyncio.Task[None]] = set()


def stored_core_queries(pool_query: dict[str, Any], cluster_id: int) -> dict[str, Any]:
    """``pool_query`` narrowed to members whose stored distance was measured
    against ``cluster_id`` (legacy docs without the reference don't match)."""
    return {
        'bool': {
            'filter': [
                pool_query,
                {'term': {'cluster_distance_cluster_id': cluster_id}},
                {'exists': {'field': 'cluster_distance'}},
            ]
        }
    }


async def _backfill(
    opensearch: Any, index: str, cluster_id: int, distances: dict[str, float]
) -> None:
    from src.services.curation.clustering.cluster_geometry import write_member_distances

    try:
        failed = await write_member_distances(
            opensearch, index=index, cluster_id=cluster_id, distances=distances
        )
        if failed:
            logger.warning('core_first_backfill_errors', cluster_id=cluster_id, n_errors=failed)
    except Exception as exc:
        logger.warning('core_first_backfill_failed', cluster_id=cluster_id, error=str(exc))
    finally:
        _BACKFILLING.discard((index, cluster_id))


def _schedule_backfill(
    opensearch: Any, index: str, cluster_id: int, distances: dict[str, float]
) -> None:
    from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET

    # Candidate clusters keep their clustering method's distances.
    if cluster_id >= RESIDUAL_CLUSTER_ID_OFFSET or (index, cluster_id) in _BACKFILLING:
        return
    _BACKFILLING.add((index, cluster_id))
    task = asyncio.get_running_loop().create_task(
        _backfill(opensearch, index, cluster_id, dict(distances))
    )
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


def with_live_distance(item: dict[str, Any], distance: float | None) -> dict[str, Any]:
    """Overlay a live centroid distance on a served item."""
    similarity = cluster_similarity(distance)
    item['cluster_distance'] = distance
    item['cluster_similarity'] = similarity
    item['cluster_is_core'] = None if similarity is None else similarity >= CORE_SIMILARITY_MIN
    return item


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
    cluster_id: int | None,
    item_filter: ItemFilter,
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

        # Pool query + exact count scoped to the embedding-bearing subset.
        pool_query, pool_count = await embedding_pool_query_and_count(
            opensearch, index, query_clause, OUTLIER_EMBEDDING_FIELD
        )
        if order == 'core_first' and pool_count > 0:
            stored = stored_core_queries(pool_query, cluster_id)
            n_stored = int(
                ((await opensearch.count(index=index, body={'query': stored})) or {}).get(
                    'count', 0
                )
            )
            if n_stored == pool_count:
                resp = await opensearch.search(
                    index=index,
                    body={
                        'query': stored,
                        'from': (page - 1) * page_size,
                        'size': page_size,
                        '_source': False,
                        'sort': [
                            {'cluster_distance': {'order': 'asc', 'unmapped_type': 'float'}},
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
            opensearch, index, pool_query, current_count=pool_count
        )
        if distances is None:
            return None
        if order == 'core_first':
            _schedule_backfill(opensearch, index, cluster_id, distances)
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
            n_unembedded=max(0, n_pool - pool_count),
            suggested_reprocess=_embed_suggestion(item_filter, n_pool - pool_count),
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
            n_unembedded=max(0, n_pool - pool_count),
            suggested_reprocess=_embed_suggestion(item_filter, n_pool - pool_count),
        )
    return None


__all__ = ['CLUSTER_ORDERS', 'ordered_crops_page', 'with_live_distance']
