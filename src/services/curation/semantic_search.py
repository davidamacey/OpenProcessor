"""Semantic search over the curation items index: by text, by a stored vector, by a picture.

``GET /curation/projects/{project}/search/similar`` ranks items by similarity to the stored vector
of one crop (no encoder). ``GET /curation/projects/{project}/search/text``
(``src.routers.curation.search``) encodes a free-text query through the text tower of the model
behind the searched space (:class:`src.clients.pe_encoder.PEEncoder`, Triton only) and hydrates
the resulting doc ids into the same response shape ``review.review_queue`` uses.

Every search runs against one *space* of the bound project's registry: the target's primary
space. The kNN query takes the space and the space key of the query vector and refuses a
mismatch (``EmbeddingSpaceMismatch``).

Gated end-to-end by ``OP_SEMANTIC_SEARCH_ENABLED`` (default on since #371, read with
``env_bool``); see ``search.py`` for the disabled-flag 400 response.

Always searches ``CurationConfig.items_index`` — the same override-aware
config every other curation router uses, so an operator flipping the
configured index name picks up the change for free.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from src.config import CurationConfig, get_curation_config
from src.config.project_context import project_api_base, run_in_executor_bound
from src.config.region_fields import RegionFields, get_region_fields
from src.core.logging import get_logger
from src.domain.embedding_spaces import (
    EmbeddingSpaceError,
    ResolvedSpace,
    Target,
    require_query_space,
)
from src.domain.embedding_state import not_embedded_clause
from src.schemas.curation.search_target import SearchTarget
from src.services.curation import review_queries
from src.services.curation.item_filter import ItemFilter, item_filter_clauses
from src.services.curation.semantic_search_box import (
    attach_match,
    box_k,
    box_source_excludes,
    box_space_source_includes,
    build_box_knn_query,
    no_box_vectors_clause,
    stored_box_entries,
)
from src.services.curation.wire import item_list_source_excludes, serialize_item
from src.services.embedding_spaces.resolve import embedding_space
from src.services.rejections import rejection
from src.utils.search_timing import search_annotate, search_span


logger = get_logger(__name__)

# How many nearest neighbors to over-fetch from the knn clause before
# paginating in Python. OpenSearch's `knn` query returns at most `k` hits
# total (not "per shard, then merged further") once filtered, so k must
# cover through the end of the requested page.
_MAX_K = 2000

_REGISTRY_TARGET = {SearchTarget.ITEM: Target.ITEM, SearchTarget.BOX: Target.BOX}


@dataclass(frozen=True)
class QueryVector:
    """A query vector and the key of the space it was made in (an encoder's model, or the
    space a stored vector was read from)."""

    values: list[float]
    space_key: str


VectorFor = Callable[[ResolvedSpace], Awaitable[QueryVector]]


def _source_excludes(fields: RegionFields) -> list[str]:
    """Never ship raw embedding vectors or class_id_history (this
    is a paginated list endpoint; history is undo-only)."""
    return item_list_source_excludes(fields)


class SemanticSearchDisabledError(RuntimeError):
    """Raised by callers that check the flag themselves; not used inside
    this module (the router owns the 400 response) but exported for any
    future caller that wants a typed signal instead of an env re-check."""


def _build_filter(
    *,
    tab: str | None,
    cluster_uid: str | None,
    date_from: str | None,
    date_to: str | None,
    max_rank: int | None,
    min_blur_ratio: float | None,
    hide_near_duplicates: bool,
    include_test: bool,
    item_filter: ItemFilter | None = None,
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
        must_not.append({'term': {'class_excluded': True}})
        if not include_test:
            must_not.append({'term': {'test_holdout': True}})
        if max_rank is not None:
            filters.append({'range': {'crop_rank_in_image': {'lte': max_rank}}})

    filters.extend(item_filter_clauses(item_filter or ItemFilter()))
    if cluster_uid is not None:
        filters.append({'term': {'cluster_uid': cluster_uid}})

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
                        {'term': {'near_dup_is_representative': True}},
                        {'bool': {'must_not': {'exists': {'field': 'near_dup_is_representative'}}}},
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
    vector_space_key: str,
    space: ResolvedSpace,
    k: int,
    filter_clause: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the OpenSearch ``knn`` query body clause over ``space``'s vector field.

    ``vector_space_key`` is the space the query vector was made in; a vector of another space
    is refused (``EmbeddingSpaceMismatch``). ``filter_clause`` is wrapped in a single
    ``bool.filter`` — the k-NN plugin accepts an arbitrary query DSL clause here (pre/post-filter
    depending on the engine's efficient-filtering support); this module doesn't second-guess
    that, it just hands the composed filter through.
    """
    require_query_space(space, vector_space_key)
    knn: dict[str, Any] = {'vector': vector, 'k': k}
    if filter_clause:
        knn['filter'] = {'bool': {'filter': filter_clause}}
    return {'knn': {space.vector_field: knn}}


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


def _filter_for(
    *,
    item_filter: ItemFilter | None,
    tab: str | None,
    cluster_uid: str | None,
    date_from: str | None,
    date_to: str | None,
    max_rank: int | None,
    min_blur_ratio: float | None,
    hide_near_duplicates: bool,
    include_test: bool,
) -> list[dict[str, Any]]:
    """The kNN ``filter`` clause both search routes share (one place, so they cannot drift)."""
    item_filter = item_filter or ItemFilter()
    return _build_filter(
        item_filter=item_filter.model_copy(update={'max_rank': None}),
        tab=tab,
        cluster_uid=cluster_uid,
        date_from=date_from,
        date_to=date_to,
        max_rank=max_rank if max_rank is not None else item_filter.max_rank,
        min_blur_ratio=min_blur_ratio,
        hide_near_duplicates=hide_near_duplicates,
        include_test=include_test,
    )


async def search_space(target: SearchTarget) -> ResolvedSpace:
    """The bound project's primary space for what ``target`` ranks."""
    return await embedding_space(_REGISTRY_TARGET[target])


async def _knn_page(
    *,
    opensearch: Any,
    vector: QueryVector,
    space: ResolvedSpace,
    filter_clause: list[dict[str, Any]],
    page: int,
    page_size: int,
    min_score: float | None,
    cfg: CurationConfig,
    fields: RegionFields,
    target: SearchTarget = SearchTarget.ITEM,
    skip_box: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Run one filtered kNN page for ``vector`` over ``space`` and hydrate it into the wire page.

    ``target=box`` ranks the per-box vectors instead; the page is still one card per item, each
    with the box that matched best. ``skip_box`` is the query's own ``(crop_id, box_id)``."""
    # `k` is the ANN candidate depth (how many nearest neighbors the
    # kNN plugin considers) — it must cover through the end of the
    # requested page, so it stays `page * page_size` (capped at _MAX_K).
    # `from`/`size` are a *separate* concern: which slice of those k
    # candidates the response actually carries; OpenSearch slices server-side,
    # so only `page_size` hits cross the wire.
    excludes = _source_excludes(fields)
    match target:
        case SearchTarget.ITEM:
            knn_query = build_knn_query(
                vector.values,
                vector_space_key=vector.space_key,
                space=space,
                k=min(_MAX_K, page * page_size),
                filter_clause=filter_clause,
            )
            unsearchable = not_embedded_clause(space.vector_field)
        case SearchTarget.BOX:
            knn_query = build_box_knn_query(
                vector.values,
                vector_space_key=vector.space_key,
                space=space,
                k=box_k(page, page_size, max_k=_MAX_K),
                filter_clause=filter_clause,
                inner_size=2 if skip_box else 1,
            )
            excludes = box_source_excludes(excludes, space)
            unsearchable = no_box_vectors_clause(space)
        case _:
            raise ValueError(f'unknown search target: {target!r}')

    body: dict[str, Any] = {
        'size': page_size,
        'from': (page - 1) * page_size,
        'query': knn_query,
        '_source': {'excludes': excludes},
    }
    if min_score is not None:
        # Top-level min_score is a generic post-scoring filter — applies
        # the same way regardless of query type, so this replaces the
        # old client-side hit filtering without changing semantics.
        body['min_score'] = min_score

    # The unembedded count shares only the filter with the kNN search, so the two run together:
    # on a filtered query the count is a full filter pass (4-8 ms at 91k items) and used to be
    # paid on top of the search (#384).
    # kNN only sees items with a vector, so "no results" must not read as "no
    # matches": say how many in-scope items it could not search.
    async def _knn() -> Any:
        with search_span('knn'):
            return await opensearch.search(index=cfg.items_index, body=body)

    async def _unembedded_count() -> Any:
        with search_span('unembedded_count'):
            return await opensearch.count(
                index=cfg.items_index,
                body={'query': {'bool': {'filter': [*filter_clause, unsearchable]}}},
            )

    with search_span('knn_page'):
        resp, unembedded = await asyncio.gather(_knn(), _unembedded_count())
    search_annotate('os_took_ms', int(resp.get('took') or 0))
    hits = (resp.get('hits') or {}).get('hits') or []
    total = int(((resp.get('hits') or {}).get('total') or {}).get('value', len(hits)))

    items: list[dict[str, Any]] = []
    with search_span('hydrate'):
        for hit in hits:
            item = _hydrate_item(hit, fields, project_api_base())
            if target is SearchTarget.BOX and not attach_match(
                item, hit, fields, space, skip=skip_box
            ):
                total = max(0, total - 1)
                continue
            items.append(item)

    return {
        'items': items,
        'total': total,
        'page': page,
        'page_size': page_size,
        'unembedded_in_scope': int((unembedded or {}).get('count', 0)),
    }


def _stored_box_vector(
    source: dict[str, Any],
    box_id: str,
    crop_id: str,
    fields: RegionFields,
    space: ResolvedSpace,
) -> list[float]:
    """The stored vector of box ``box_id`` of a crop in ``space``; 404 ``box_not_found`` / 409
    ``box_not_embedded`` when the crop has no such box / the box has no vector."""
    known = {b.get('box_id') for b in source.get(fields.boxes) or [] if isinstance(b, dict)}
    leaf = space.vector_field.rsplit('.', 1)[-1]
    for entry in stored_box_entries(source, space):
        if entry.get('box_id') != box_id:
            continue
        vector = entry.get(leaf)
        if isinstance(vector, list) and vector:
            return vector
        known.add(box_id)
    if box_id in known:
        raise rejection(409, 'box_not_embedded', f'box {box_id} of crop {crop_id} has no vector')
    raise rejection(404, 'box_not_found', f"no box '{box_id}' on crop '{crop_id}'")


def _text_vector_for(pe_encoder: Any, executor: Any, query: str) -> VectorFor:
    """Encode ``query`` for a space, refusing (422) a space whose text model this API does not
    serve (only the PE text tower is encodable here)."""

    async def vector_for(space: ResolvedSpace) -> QueryVector:
        if space.triton_text_model != pe_encoder.text_model:
            raise EmbeddingSpaceError(
                'embedding_model_unavailable',
                f'space {space.space_key} needs the text model {space.triton_text_model!r}; '
                f'this API encodes text with {pe_encoder.text_model!r}',
                status=422,
            )
        loop = asyncio.get_running_loop()
        with search_span('encode'):
            vectors = await run_in_executor_bound(loop, executor, pe_encoder.encode_text, [query])
        return QueryVector(vectors[0].tolist(), space.space_key)

    return vector_for


async def semantic_text_search(
    *,
    opensearch: Any,
    pe_encoder: Any,
    executor: Any,
    query: str,
    page: int,
    page_size: int,
    cluster_uid: str | None = None,
    tab: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    max_rank: int | None = None,
    min_blur_ratio: float | None = None,
    hide_near_duplicates: bool = False,
    min_score: float | None = None,
    include_test: bool = False,
    item_filter: ItemFilter | None = None,
    target: SearchTarget = SearchTarget.ITEM,
    config: CurationConfig | None = None,
    fields: RegionFields | None = None,
) -> dict[str, Any]:
    """Run one semantic text search and return ``{items, total, page, page_size}``.

    ``pe_encoder.encode_text`` is a blocking call (a synchronous Triton gRPC request) — it is
    **never** invoked inline on the event loop; the caller-supplied ``executor`` (the app's
    shared ``ThreadPoolExecutor``, same one every other CPU-bound curation path uses) runs it
    via ``loop.run_in_executor``.
    """
    if not query or not query.strip():
        return {
            'items': [],
            'total': 0,
            'page': page,
            'page_size': page_size,
            'unembedded_in_scope': 0,
        }

    cfg = config or get_curation_config()
    region_fields = fields or get_region_fields()
    space = await search_space(target)
    vector = await _text_vector_for(pe_encoder, executor, query)(space)

    with search_span('filter_build'):
        filter_clause = _filter_for(
            item_filter=item_filter,
            tab=tab,
            cluster_uid=cluster_uid,
            date_from=date_from,
            date_to=date_to,
            max_rank=max_rank,
            min_blur_ratio=min_blur_ratio,
            hide_near_duplicates=hide_near_duplicates,
            include_test=include_test,
        )
    return await _knn_page(
        opensearch=opensearch,
        vector=vector,
        space=space,
        filter_clause=filter_clause,
        page=page,
        page_size=page_size,
        min_score=min_score,
        cfg=cfg,
        fields=region_fields,
        target=target,
    )


async def _source_space(box_id: str | None, space: ResolvedSpace) -> ResolvedSpace:
    """The space the stored query vector is read from: the same-keyed space of what the query
    names (the crop's item vector, or the box's vector with ``box_id``)."""
    return await embedding_space(Target.ITEM if box_id is None else Target.BOX, space.space_key)


async def semantic_similar_search(
    *,
    opensearch: Any,
    crop_id: str,
    page: int,
    page_size: int,
    box_id: str | None = None,
    target: SearchTarget = SearchTarget.ITEM,
    exclude_self: bool = True,
    cluster_uid: str | None = None,
    tab: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    max_rank: int | None = None,
    min_blur_ratio: float | None = None,
    hide_near_duplicates: bool = False,
    min_score: float | None = None,
    include_test: bool = False,
    item_filter: ItemFilter | None = None,
    config: CurationConfig | None = None,
    fields: RegionFields | None = None,
) -> dict[str, Any]:
    """Rank items by similarity to the stored vector of ``crop_id`` (same page and filters
    as :func:`semantic_text_search`; nothing is encoded).

    With ``box_id`` the query vector is that box's stored vector instead of the crop's. Refuses
    404 ``not_found`` for a crop the bound project's items index lacks (``box_not_found`` for
    a box it lacks) and 409 ``crop_not_embedded`` (``box_not_embedded``) for one without a
    vector, so a missing source never reads as "no neighbours".

    ``exclude_self`` drops the source crop; with ``box_id`` on ``target=box`` it drops only the
    source box, so the crop's other boxes can still match.
    """
    cfg = config or get_curation_config()
    region_fields = fields or get_region_fields()
    space = await search_space(target)
    source_space = await _source_space(box_id, space)
    includes = [
        f'{region_fields.boxes}.box_id',
        *(
            box_space_source_includes(source_space)
            if box_id is not None
            else [source_space.vector_field, source_space.state_field or source_space.vector_field]
        ),
    ]

    with search_span('source_vector'):
        resp = await opensearch.search(
            index=cfg.items_index,
            body={
                'size': 1,
                'query': {'ids': {'values': [crop_id]}},
                '_source': {'includes': includes},
            },
        )
    hits = (resp.get('hits') or {}).get('hits') or []
    if not hits:
        raise rejection(404, 'not_found', f"no crop '{crop_id}' in this project")
    source = hits[0].get('_source') or {}
    if box_id is not None:
        values = _stored_box_vector(source, box_id, crop_id, region_fields, source_space)
    else:
        stored = source.get(source_space.vector_field)
        if not isinstance(stored, list) or not stored:
            state = source.get(source_space.state_field or '') or 'unknown'
            raise rejection(409, 'crop_not_embedded', f'crop {crop_id} has no vector ({state})')
        values = stored
    vector = QueryVector(values, source_space.space_key)
    skip_box = None

    with search_span('filter_build'):
        filter_clause = _filter_for(
            item_filter=item_filter,
            tab=tab,
            cluster_uid=cluster_uid,
            date_from=date_from,
            date_to=date_to,
            max_rank=max_rank,
            min_blur_ratio=min_blur_ratio,
            hide_near_duplicates=hide_near_duplicates,
            include_test=include_test,
        )
    if exclude_self:
        if box_id is not None and target is SearchTarget.BOX:
            skip_box = (crop_id, box_id)
        else:
            filter_clause.append({'bool': {'must_not': [{'ids': {'values': [crop_id]}}]}})
    return await _knn_page(
        opensearch=opensearch,
        vector=vector,
        space=space,
        filter_clause=filter_clause,
        page=page,
        page_size=page_size,
        min_score=min_score,
        cfg=cfg,
        fields=region_fields,
        target=target,
        skip_box=skip_box,
    )


__all__ = [
    'QueryVector',
    'SemanticSearchDisabledError',
    'build_knn_query',
    'no_box_vectors_clause',
    'search_space',
    'semantic_similar_search',
    'semantic_text_search',
]
