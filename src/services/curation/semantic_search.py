"""Semantic search over the curation items index: by text, by a stored vector, by a picture.

``GET /curation/projects/{project}/search/similar`` ranks items by similarity to the stored vector
of one crop (no encoder). ``GET /curation/projects/{project}/search/text``
(``src.routers.curation.search``) encodes a free-text query through the text tower of the model
behind the searched space (:class:`src.clients.pe_encoder.PEEncoder`, Triton only) and hydrates
the resulting doc ids into the same response shape ``review.review_queue`` uses.

Every search runs against one *space* of the bound project's registry: the target's primary
space unless ``space`` names another live one. ``fuse`` adds more spaces and merges the
per-space rankings by reciprocal rank fusion (RRF); an item keeps the cosine ``semantic_score``
of the first space that found it, and ``score_space`` names that space. The kNN query takes the
space and the space key of the query vector and refuses a mismatch (``EmbeddingSpaceMismatch``).
The kNN page of every space and the unembedded count go out as one ``_msearch`` round trip.

Gated end-to-end by ``OP_SEMANTIC_SEARCH_ENABLED`` (default on since #371, read with
``env_bool``); see ``search.py`` for the disabled-flag 400 response.

Always searches ``CurationConfig.items_index`` — the same override-aware
config every other curation router uses, so an operator flipping the
configured index name picks up the change for free.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.config import CurationConfig, get_curation_config
from src.config.project_context import run_in_executor_bound
from src.config.region_fields import RegionFields, get_region_fields
from src.core.logging import get_logger
from src.domain.embedding_spaces import EmbeddingSpaceError, ResolvedSpace, Target
from src.schemas.curation.search_target import SearchTarget
from src.services.curation import review_queries
from src.services.curation.item_filter import ItemFilter, item_filter_clauses
from src.services.curation.semantic_search_box import box_space_source_includes, stored_box_entries
from src.services.curation.semantic_search_run import (
    QueryVector,
    VectorFor,
    resolve_search_spaces,
    run_search,
)
from src.services.embedding_spaces.readers import item_write_fields
from src.services.embedding_spaces.resolve import embedding_space
from src.services.rejections import rejection
from src.utils.search_timing import search_span


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.embedding_spaces.readers import WriteFields


logger = get_logger(__name__)


class SemanticSearchDisabledError(RuntimeError):
    """Raised by callers that check the flag themselves; not used inside
    this module (the router owns the 400 response) but exported for any
    future caller that wants a typed signal instead of an env re-check."""


def _build_filter(
    *,
    embedding: WriteFields,
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
            tab,
            include_test=include_test,
            text=None,
            max_rank=max_rank,
            vector_field=embedding.vector_field,
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

    filters.extend(item_filter_clauses(item_filter or ItemFilter(), embedding=embedding))
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


def _filter_for(
    *,
    embedding: WriteFields,
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
        embedding=embedding,
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
    space_key: str | None = None,
    fuse: Sequence[str] = (),
    config: CurationConfig | None = None,
    fields: RegionFields | None = None,
) -> dict[str, Any]:
    """Run one semantic text search and return ``{items, total, page, page_size, space, fuse}``.

    ``pe_encoder.encode_text`` is a blocking call (a synchronous Triton gRPC request) — it is
    **never** invoked inline on the event loop; the caller-supplied ``executor`` (the app's
    shared ``ThreadPoolExecutor``, same one every other CPU-bound curation path uses) runs it
    via ``loop.run_in_executor``.
    """
    cfg = config or get_curation_config()
    region_fields = fields or get_region_fields()
    spaces = await resolve_search_spaces(target, space_key, fuse)
    if not query or not query.strip():
        return {
            'items': [],
            'total': 0,
            'page': page,
            'page_size': page_size,
            'unembedded_in_scope': 0,
            'space': spaces[0].space_key,
            'fuse': [s.space_key for s in spaces[1:]],
        }

    with search_span('filter_build'):
        filter_clause = _filter_for(
            embedding=await item_write_fields(),
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
    return await run_search(
        opensearch=opensearch,
        spaces=spaces,
        vector_for=_text_vector_for(pe_encoder, executor, query),
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
    space_key: str | None = None,
    fuse: Sequence[str] = (),
    config: CurationConfig | None = None,
    fields: RegionFields | None = None,
) -> dict[str, Any]:
    """Rank items by similarity to the stored vector of ``crop_id`` (same page and filters
    as :func:`semantic_text_search`; nothing is encoded).

    With ``box_id`` the query vector is that box's stored vector instead of the crop's. The
    vector of each searched space is the one stored under that space's key. Refuses 404
    ``not_found`` for a crop the bound project's items index lacks (``box_not_found`` for a box
    it lacks) and 409 ``crop_not_embedded`` (``box_not_embedded``) for one without a vector in a
    searched space, so a missing source never reads as "no neighbours".

    ``exclude_self`` drops the source crop; with ``box_id`` on ``target=box`` it drops only the
    source box, so the crop's other boxes can still match.
    """
    cfg = config or get_curation_config()
    region_fields = fields or get_region_fields()
    spaces = await resolve_search_spaces(target, space_key, fuse)
    sources = {s.space_key: await _source_space(box_id, s) for s in spaces}
    includes = [f'{region_fields.boxes}.box_id']
    for source_space in sources.values():
        if box_id is None:
            includes.extend(
                [
                    source_space.vector_field,
                    *([source_space.state_field] if source_space.state_field else []),
                ]
            )
        else:
            includes.extend(box_space_source_includes(source_space))

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

    async def vector_for(space: ResolvedSpace) -> QueryVector:
        source_space = sources[space.space_key]
        if box_id is not None:
            values = _stored_box_vector(source, box_id, crop_id, region_fields, source_space)
        else:
            stored = source.get(source_space.vector_field)
            if not isinstance(stored, list) or not stored:
                state = source.get(source_space.state_field or '') or 'unknown'
                raise rejection(409, 'crop_not_embedded', f'crop {crop_id} has no vector ({state})')
            values = stored
        return QueryVector(values, source_space.space_key)

    skip_box = None
    with search_span('filter_build'):
        filter_clause = _filter_for(
            embedding=await item_write_fields(),
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
    return await run_search(
        opensearch=opensearch,
        spaces=spaces,
        vector_for=vector_for,
        filter_clause=filter_clause,
        page=page,
        page_size=page_size,
        min_score=min_score,
        cfg=cfg,
        fields=region_fields,
        target=target,
        skip_box=skip_box,
    )


__all__ = ['SemanticSearchDisabledError', 'semantic_similar_search', 'semantic_text_search']
