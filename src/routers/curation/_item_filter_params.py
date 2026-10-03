"""The query parameters of the shared item filter, declared once.

Every list, search and stats route that filters items takes
:data:`ItemFilterQuery` and builds its query with
:func:`~src.services.curation.item_filter.item_filter_clauses`, so one set of
parameter names and one meaning of each (class by NAME, confidence band, box
size, N largest per image, origin, embedding state, review status) holds
across the API. Routes keep their own route-specific parameters on top.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query

from src.services.curation.embedding_state import (
    EmbeddingState,  # noqa: TC001 - FastAPI resolves it
)
from src.services.curation.item_filter import ItemFilter, Origin, ReviewStatus


def item_filter_query(
    class_name: Annotated[
        list[str] | None,
        Query(
            description=(
                "Class by NAME, repeatable (OR): matches an item's class name or the "
                "detector's own label; case, spaces and hyphens are normalized "
                "('traffic light' = 'traffic_light')."
            )
        ),
    ] = None,
    exclude_class_name: Annotated[
        list[str] | None, Query(description='Hide these class names (same matching).')
    ] = None,
    conf_min: Annotated[
        float | None, Query(ge=0.0, le=1.0, description='Inclusive confidence band, lower bound.')
    ] = None,
    conf_max: Annotated[
        float | None, Query(ge=0.0, le=1.0, description='Inclusive confidence band, upper bound.')
    ] = None,
    min_area: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description='Box area as a fraction of its image, lower bound.'),
    ] = None,
    max_area: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description='Box area as a fraction of its image, upper bound.'),
    ] = None,
    max_rank: Annotated[
        int | None,
        Query(ge=1, description='The N largest boxes per image (crop_rank_in_image <= N).'),
    ] = None,
    origin: Annotated[
        list[Origin] | None,
        Query(
            description=(
                'How the item came to exist, repeatable (OR): detector, sam3, human or import.'
            )
        ),
    ] = None,
    embedding_state: Annotated[
        list[EmbeddingState] | None,
        Query(description='embedded (has a vector), not_selected, deferred or failed; repeatable.'),
    ] = None,
    review_status: Annotated[
        list[ReviewStatus] | None,
        Query(description='pending, validated, dismissed or excluded; repeatable.'),
    ] = None,
) -> ItemFilter:
    return ItemFilter(
        class_names=class_name or [],
        exclude_class_names=exclude_class_name or [],
        conf_min=conf_min,
        conf_max=conf_max,
        min_area=min_area,
        max_area=max_area,
        max_rank=max_rank,
        origin=origin or [],
        embedding_state=embedding_state or [],
        review_status=review_status or [],
    )


ItemFilterQuery = Annotated[ItemFilter, Depends(item_filter_query)]

__all__ = ['ItemFilterQuery', 'item_filter_query']
