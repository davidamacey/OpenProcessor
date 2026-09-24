"""Semantic text search over the curation items index.

``GET /curation/search/text`` (``src.routers.curation.search``) encodes a
free-text query through the PE-Core-L14-336 **text** tower
(:class:`src.clients.pe_encoder.PEEncoder`), builds an OpenSearch ``knn``
query against the item's PE-Core **image**-tower embedding
(``pe_embedding`` — same checkpoint, same L2-normalized 1024-d space, see
``pe_encoder.py``'s module docstring), and hydrates the resulting doc ids
into the same response shape ``review.review_queue`` uses.

Gated end-to-end by ``OP_SEMANTIC_SEARCH_ENABLED`` (default off, inline
``os.getenv`` — same house convention ``scores.py``/``select.py`` use for
their own flags); see ``search.py`` for the disabled-flag 400 response.

Always searches ``CurationConfig.items_index`` — the same override-aware
config every other curation router uses, so an operator flipping the
configured index name picks up the change for free.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.config import CurationConfig, get_curation_config
from src.config.region_fields import RegionFields, get_region_fields
from src.core.logging import get_logger
from src.services.curation import review_queries
from src.services.curation.wire import item_source_excludes, serialize_item


logger = get_logger(__name__)

DEFAULT_KNN_FIELD = 'pe_embedding'

# How many nearest neighbors to over-fetch from the knn clause before
# paginating in Python. OpenSearch's `knn` query returns at most `k` hits
# total (not "per shard, then merged further") once filtered, so k must
# cover through the end of the requested page.
_MAX_K = 2000


def _source_excludes(fields: RegionFields) -> list[str]:
    """Never ship raw embedding vectors — same list every item endpoint uses."""
    return item_source_excludes(fields)


class SemanticSearchDisabledError(RuntimeError):
    """Raised by callers that check the flag themselves; not used inside
    this module (the router owns the 400 response) but exported for any
    future caller that wants a typed signal instead of an env re-check."""


def _build_filter(
    *,
    tab: str | None,
    class_id: int | None,
    cluster_id: int | None,
    date_from: str | None,
    date_to: str | None,
    max_rank: int | None,
    min_blur_ratio: float | None,
    hide_near_duplicates: bool,
    include_test: bool,
) -> list[dict[str, Any]]:
    """Compose the kNN query's ``filter`` clause.

    Reuses :func:`review_queries.build_tab_query` for the ``tab`` cohort
    (same ``must``/``must_not`` a human would see on ``/review``) rather
    than re-deriving tab semantics here — a semantic-search-within-a-tab
    result set stays consistent with what that tab's plain listing shows.
    """
    filters: list[dict[str, Any]] = []
    must_not: list[dict[str, Any]] = []

    if tab:
        tab_must, tab_must_not, _reason = review_queries.build_tab_query(
            tab, include_test=include_test, text=None, max_rank=max_rank
        )
        filters.extend(tab_must)
        must_not.extend(tab_must_not)
    else:
        # No tab selected — still honor the review-queue invariants a bare
        # semantic search over the whole pool should respect: validated
        # items and permanently-dismissed items stay out, same as every
        # /review tab (review_queries.build_tab_query's own must_not).
        must_not.append({'term': {'class_validated': True}})
        must_not.append({'exists': {'field': 'review_dismissed_at'}})
        if not include_test:
            must_not.append({'term': {'test_holdout': True}})
        if max_rank is not None:
            filters.append({'range': {'crop_rank_in_image': {'lte': max_rank}}})

    if class_id is not None:
        filters.append({'term': {'class_id': class_id}})
    if cluster_id is not None:
        filters.append({'term': {'cluster_id': cluster_id}})

    if date_from is not None or date_to is not None:
        date_range: dict[str, Any] = {}
        if date_from is not None:
            date_range['gte'] = date_from
        if date_to is not None:
            date_range['lte'] = date_to
        filters.append({'range': {'created_at': date_range}})

    # Null-safe clarity slider — same "missing field never hides the item"
    # pattern review.review_queue uses for min_blur_ratio.
    if min_blur_ratio is not None:
        filters.append(
            {
                'bool': {
                    'should': [
                        {'range': {'blur_lap_ratio': {'gte': min_blur_ratio}}},
                        {'bool': {'must_not': {'exists': {'field': 'blur_lap_ratio'}}}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )

    if hide_near_duplicates:
        filters.append(
            {
                'bool': {
                    'should': [
                        {'term': {'dup_is_representative': True}},
                        {'bool': {'must_not': {'exists': {'field': 'dup_is_representative'}}}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )

    if must_not:
        filters.append({'bool': {'must_not': must_not}})

    return filters


def build_knn_query(
    vector: list[float],
    *,
    k: int,
    filter_clause: list[dict[str, Any]],
    field: str = DEFAULT_KNN_FIELD,
) -> dict[str, Any]:
    """Build the OpenSearch ``knn`` query body clause.

    ``filter_clause`` is wrapped in a single ``bool.filter`` — the k-NN
    plugin accepts an arbitrary query DSL clause here (pre/post-filter
    depending on the ``faiss``/HNSW engine's efficient-filtering support);
    this module doesn't second-guess that, it just hands the composed
    filter through.
    """
    knn: dict[str, Any] = {'vector': vector, 'k': k}
    if filter_clause:
        knn['filter'] = {'bool': {'filter': filter_clause}}
    return {'knn': {field: knn}}


def _hydrate_item(hit: dict[str, Any], fields: RegionFields, api_prefix: str) -> dict[str, Any]:
    """Project one OpenSearch hit onto the shared wire item (same keys as
    ``/crops`` and ``/review``) plus a ``semantic_score`` (the raw kNN
    cosine-similarity ``_score``) so the UI can render a relevance chip.
    """
    item = serialize_item(
        hit.get('_source') or {}, hit.get('_id', ''), storage=fields, api_prefix=api_prefix
    )
    item['semantic_score'] = hit.get('_score')
    return item


async def semantic_text_search(
    *,
    opensearch: Any,
    pe_encoder: Any,
    executor: Any,
    query: str,
    page: int,
    page_size: int,
    class_id: int | None = None,
    cluster_id: int | None = None,
    tab: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    max_rank: int | None = None,
    min_blur_ratio: float | None = None,
    hide_near_duplicates: bool = False,
    min_score: float | None = None,
    include_test: bool = False,
    config: CurationConfig | None = None,
    fields: RegionFields | None = None,
) -> dict[str, Any]:
    """Run one semantic text search and return ``{items, total, page, page_size}``.

    ``pe_encoder.encode_text`` is a blocking, CPU/GPU-bound PyTorch call —
    it is **never** invoked inline on the event loop; the caller-supplied
    ``executor`` (the app's shared ``ThreadPoolExecutor``, same one every
    other CPU-bound curation path uses) runs it via
    ``loop.run_in_executor``.
    """
    if not query or not query.strip():
        return {'items': [], 'total': 0, 'page': page, 'page_size': page_size}

    cfg = config or get_curation_config()
    region_fields = fields or get_region_fields()

    loop = asyncio.get_running_loop()
    vectors = await loop.run_in_executor(executor, pe_encoder.encode_text, [query])
    vector = vectors[0].tolist()

    filter_clause = _build_filter(
        tab=tab,
        class_id=class_id,
        cluster_id=cluster_id,
        date_from=date_from,
        date_to=date_to,
        max_rank=max_rank,
        min_blur_ratio=min_blur_ratio,
        hide_near_duplicates=hide_near_duplicates,
        include_test=include_test,
    )

    # Over-fetch through the end of the requested page (kNN doesn't support
    # from/size pagination server-side — it returns the top-k globally).
    k = min(_MAX_K, page * page_size)
    knn_query = build_knn_query(vector, k=k, filter_clause=filter_clause)

    body = {
        'size': k,
        'query': knn_query,
        '_source': {'excludes': _source_excludes(region_fields)},
    }

    resp = await opensearch.search(index=cfg.items_index, body=body)
    hits = (resp.get('hits') or {}).get('hits') or []

    if min_score is not None:
        hits = [h for h in hits if (h.get('_score') is None or h.get('_score', 0.0) >= min_score)]

    total = len(hits)
    start = (page - 1) * page_size
    page_hits = hits[start : start + page_size]
    items = [_hydrate_item(h, region_fields, cfg.api_prefix) for h in page_hits]

    return {'items': items, 'total': total, 'page': page, 'page_size': page_size}


__all__ = [
    'DEFAULT_KNN_FIELD',
    'SemanticSearchDisabledError',
    'build_knn_query',
    'semantic_text_search',
]
