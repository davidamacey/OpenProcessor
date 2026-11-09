"""Run a semantic search over one or several spaces: the kNN query, the one ``_msearch``, fusion.

The building blocks :mod:`src.services.curation.semantic_search` composes with its filters and
its query sources (text, a stored vector, a picture): the kNN query over a space (refusing a
query vector made in another space), the page body and the unembedded count, the single
``_msearch`` round trip, reciprocal rank fusion (RRF) and the wire page.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config.project_context import project_api_base
from src.domain.embedding_spaces import ResolvedSpace, Target, require_query_space
from src.domain.embedding_state import not_embedded_clause
from src.schemas.curation.search_target import SearchTarget
from src.services.curation.semantic_search_box import (
    attach_match,
    box_k,
    box_source_excludes,
    build_box_knn_query,
    no_box_vectors_clause,
)
from src.services.curation.wire import item_list_source_excludes, serialize_item
from src.services.embedding_spaces.readers import item_state_fields, vector_source_excludes
from src.services.embedding_spaces.resolve import embedding_space
from src.utils.search_timing import search_annotate, search_span


if TYPE_CHECKING:
    from src.config import CurationConfig
    from src.config.region_fields import RegionFields


# The constant of reciprocal rank fusion: a hit at rank r in one list adds 1 / (RRF_RANK_CONSTANT
# + r). 60 is the value of the original method and the default of OpenSearch's RRF score ranker
# (https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/).
RRF_RANK_CONSTANT = 60

# How many nearest neighbors to over-fetch from the knn clause before
# paginating in Python. OpenSearch's `knn` query returns at most `k` hits
# total (not "per shard, then merged further") once filtered, so k must
# cover through the end of the requested page.
MAX_K = 2000

_REGISTRY_TARGET = {SearchTarget.ITEM: Target.ITEM, SearchTarget.BOX: Target.BOX}


@dataclass(frozen=True)
class QueryVector:
    """A query vector and the key of the space it was made in (an encoder's model, or the
    space a stored vector was read from)."""

    values: list[float]
    space_key: str


VectorFor = Callable[[ResolvedSpace], Awaitable[QueryVector]]


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


def hydrate_item(
    hit: dict[str, Any],
    fields: RegionFields,
    api_prefix: str,
    space: ResolvedSpace,
    state_fields: Mapping[str, str],
) -> dict[str, Any]:
    """Project one OpenSearch hit onto the shared wire item (same keys as
    ``/crops`` and ``/review``) plus a ``semantic_score`` (the raw kNN
    cosine-similarity ``_score``) so the UI can render a relevance chip.
    """
    item = serialize_item(
        hit.get('_source') or {},
        hit.get('_id', ''),
        state_fields=state_fields,
        storage=fields,
        api_prefix=api_prefix,
    )
    item['semantic_score'] = hit.get('_score')
    item['score_space'] = space.space_key
    return item


def _count_body(
    filter_clause: list[dict[str, Any]], unsearchable: dict[str, Any]
) -> dict[str, Any]:
    """The in-scope items a space cannot search: a size-0 search with an exact total (``_msearch``
    has no ``_count``)."""
    return {
        'size': 0,
        'track_total_hits': True,
        'query': {'bool': {'filter': [*filter_clause, unsearchable]}},
    }


def _total(response: dict[str, Any]) -> int:
    return int(((response.get('hits') or {}).get('total') or {}).get('value', 0))


async def _msearch(
    opensearch: Any, index: str, bodies: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One ``_msearch`` for ``bodies`` against ``index``; any failed entry fails the call
    (OpenSearch answers 200 with a per-entry ``error``).

    OpenSearch multi-search: https://docs.opensearch.org/latest/api-reference/multi-search/"""
    payload: list[dict[str, Any]] = []
    for body in bodies:
        payload.extend(({'index': index}, body))
    with search_span('knn_page'):
        resp = await opensearch.msearch(body=payload)
    responses: list[dict[str, Any]] = resp.get('responses') or []
    if len(responses) != len(bodies):
        raise RuntimeError(f'_msearch answered {len(responses)} of {len(bodies)} searches')
    for entry in responses:
        if entry.get('error'):
            raise RuntimeError(f'_msearch entry failed: {entry["error"]}')
    return responses


async def resolve_search_spaces(
    target: SearchTarget, space_key: str | None, fuse: Sequence[str]
) -> list[ResolvedSpace]:
    """The searched space (the target's primary unless ``space_key`` names another live one),
    then the live spaces to fuse in; 404 ``embedding_space_not_found`` for an unknown key."""
    registry_target = _REGISTRY_TARGET[target]
    primary = await embedding_space(registry_target, space_key)
    others = [
        await embedding_space(registry_target, key)
        for key in dict.fromkeys(fuse)
        if key != primary.space_key
    ]
    return [primary, *others]


def _search_body(
    *,
    vector: QueryVector,
    space: ResolvedSpace,
    filter_clause: list[dict[str, Any]],
    from_: int,
    size: int,
    depth: int,
    min_score: float | None,
    vector_excludes: Sequence[str],
    target: SearchTarget,
    skip_box: tuple[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The kNN search body over ``space`` and the clause for the items it cannot search.

    ``depth`` is the ANN candidate depth (how many nearest neighbors the kNN plugin considers):
    it must cover through the end of the requested page. ``from_``/``size`` are a separate
    concern, which slice of those candidates the response carries; OpenSearch slices
    server-side, so only that slice crosses the wire."""
    # A paginated list endpoint: never ship raw vectors or class_id_history (undo-only).
    excludes = item_list_source_excludes(vector_excludes)
    match target:
        case SearchTarget.ITEM:
            knn_query = build_knn_query(
                vector.values,
                vector_space_key=vector.space_key,
                space=space,
                k=depth,
                filter_clause=filter_clause,
            )
            unsearchable = not_embedded_clause(space.vector_field)
        case SearchTarget.BOX:
            knn_query = build_box_knn_query(
                vector.values,
                vector_space_key=vector.space_key,
                space=space,
                k=box_k(depth, 1, max_k=MAX_K),
                filter_clause=filter_clause,
                inner_size=2 if skip_box else 1,
            )
            excludes = box_source_excludes(excludes, space)
            unsearchable = no_box_vectors_clause(space)
        case _:
            raise ValueError(f'unknown search target: {target!r}')
    body: dict[str, Any] = {
        'size': size,
        'from': from_,
        'query': knn_query,
        '_source': {'excludes': excludes},
    }
    if min_score is not None:
        # Top-level min_score is a generic post-scoring filter: it applies the same way
        # whatever the query type.
        body['min_score'] = min_score
    return body, unsearchable


def _wire_page(
    ranked: Sequence[tuple[dict[str, Any], ResolvedSpace]],
    *,
    total: int,
    page: int,
    page_size: int,
    unembedded: int,
    fields: RegionFields,
    state_fields: Mapping[str, str],
    target: SearchTarget,
    skip_box: tuple[str, str] | None,
    spaces: Sequence[ResolvedSpace],
) -> dict[str, Any]:
    """Hydrate ``ranked`` ``(hit, space of the hit)`` pairs into the response page."""
    items: list[dict[str, Any]] = []
    with search_span('hydrate'):
        for hit, space in ranked:
            item = hydrate_item(hit, fields, project_api_base(), space, state_fields)
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
        'unembedded_in_scope': unembedded,
        'space': spaces[0].space_key,
        'fuse': [s.space_key for s in spaces[1:]],
    }


def rrf_merge(
    ranked: Sequence[tuple[ResolvedSpace, list[dict[str, Any]]]],
) -> list[tuple[dict[str, Any], ResolvedSpace, float]]:
    """Reciprocal rank fusion of per-space rankings: ``(hit, space of the hit, fused score)``,
    best first. A document keeps the hit of the first list that has it; ties break on the id."""
    fused: dict[str, float] = {}
    first: dict[str, tuple[dict[str, Any], ResolvedSpace]] = {}
    for space, hits in ranked:
        for rank, hit in enumerate(hits, start=1):
            doc_id = str(hit['_id'])
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_RANK_CONSTANT + rank)
            first.setdefault(doc_id, (hit, space))
    return [(*first[i], fused[i]) for i in sorted(fused, key=lambda i: (-fused[i], i))]


async def run_search(
    *,
    opensearch: Any,
    spaces: Sequence[ResolvedSpace],
    vector_for: VectorFor,
    filter_clause: list[dict[str, Any]],
    page: int,
    page_size: int,
    min_score: float | None,
    cfg: CurationConfig,
    fields: RegionFields,
    target: SearchTarget,
    skip_box: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Search ``spaces`` (the first is the searched space, the rest are fused in by RRF) and
    return the wire page. Every space's kNN page and the primary space's unembedded count travel
    in one ``_msearch``."""
    queries = [(space, await vector_for(space)) for space in spaces]
    fused = len(queries) > 1
    depth = min(MAX_K, page * page_size)
    bodies: list[dict[str, Any]] = []
    unsearchable: dict[str, Any] = {}
    vector_excludes = await vector_source_excludes()
    for space, vector in queries:
        body, clause = _search_body(
            vector=vector,
            space=space,
            filter_clause=filter_clause,
            from_=0 if fused else (page - 1) * page_size,
            size=depth if fused else page_size,
            depth=depth,
            min_score=min_score,
            vector_excludes=vector_excludes,
            target=target,
            skip_box=skip_box,
        )
        bodies.append(body)
        unsearchable = unsearchable or clause
    responses = await _msearch(
        opensearch, cfg.items_index, [*bodies, _count_body(filter_clause, unsearchable)]
    )
    *searches, counted = responses
    search_annotate('os_took_ms', int(searches[0].get('took') or 0))
    per_space = [
        (space, (resp.get('hits') or {}).get('hits') or [])
        for (space, _vector), resp in zip(queries, searches, strict=True)
    ]
    if not fused:
        hits = per_space[0][1]
        total = int(((searches[0].get('hits') or {}).get('total') or {}).get('value', len(hits)))
        ranked = [(hit, spaces[0]) for hit in hits]
    else:
        merged = rrf_merge(per_space)
        total = len(merged)
        start = (page - 1) * page_size
        ranked = [(hit, space) for hit, space, _score in merged[start : start + page_size]]
    return _wire_page(
        ranked,
        total=total,
        page=page,
        page_size=page_size,
        unembedded=_total(counted),
        fields=fields,
        state_fields=await item_state_fields(),
        target=target,
        skip_box=skip_box,
        spaces=spaces,
    )
