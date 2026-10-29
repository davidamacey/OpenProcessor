"""Shared builders for the ``GET /clusters`` tests: aggregation buckets and a call helper."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from src.services.curation.cluster_cards import (
    CLUSTER_REVIEW_TABS,
    ClusterListParams,
    list_cluster_cards,
    review_agg_name,
)


def bucket(
    cid: int,
    size: int,
    classes: list[tuple[str, int]] | None = None,
    *,
    labelled: int | None = None,
    validated: int = 0,
    detector: list[tuple[str, int]] | None = None,
    detector_conf: float | None = None,
    review: dict[str, int] | None = None,
    updated: str | None = None,
) -> dict[str, Any]:
    """One ``cluster_id`` terms bucket as OpenSearch returns it."""
    classes = classes if classes is not None else [('van', size)]
    detector = detector or []
    out: dict[str, Any] = {
        'key': cid,
        'doc_count': size,
        'top_class': {'buckets': [{'key': k, 'doc_count': n} for k, n in classes[:3]]},
        'labelled': {'doc_count': sum(n for _k, n in classes) if labelled is None else labelled},
        'validated': {'doc_count': validated},
        'subclusters': {'value': 0},
        'latest_update': {'value_as_string': updated} if updated else {},
        'detector_top': {'buckets': [{'key': k, 'doc_count': n} for k, n in detector[:1]]},
        'detector_labelled': {'doc_count': sum(n for _k, n in detector)},
        'detector_conf': {'value': detector_conf},
    }
    for tab in CLUSTER_REVIEW_TABS:
        out[review_agg_name(tab)] = {'doc_count': (review or {}).get(tab, 0)}
    return out


def params(**over: Any) -> ClusterListParams:
    base: dict[str, Any] = {'per_cluster': 0, 'max_clusters': 100}
    base.update(over)
    return ClusterListParams(**base)


async def list_cards(os_client: Any, **over: Any) -> dict[str, Any]:
    """The response as the wire dict."""
    return dict(await list_cluster_cards(os_client, params(**over)))


def os_returning(*buckets: dict[str, Any]) -> AsyncMock:
    client = AsyncMock()
    client.search = AsyncMock(
        return_value={'aggregations': {'clusters': {'buckets': list(buckets)}}}
    )
    return client


async def cards_by_id(*buckets: dict[str, Any], **over: Any) -> dict[int, dict[str, Any]]:
    resp = await list_cards(os_returning(*buckets), **over)
    return {c['cluster_id']: c for c in resp['items']}


__all__ = ['bucket', 'cards_by_id', 'list_cards', 'os_returning', 'params']
