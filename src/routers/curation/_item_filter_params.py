"""The query parameters of the shared item filter, declared once.

Every list, search and stats route that filters items takes
:data:`ItemFilterQuery` and builds its query with
:func:`~src.services.curation.item_filter.item_filter_clauses`, so one set of
parameter names and one meaning of each (class by NAME, confidence band, box
size, N largest per image, origin, embedding state, review status) holds
across the API. Routes keep their own route-specific parameters on top.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Query
from fastapi.exceptions import RequestValidationError

from src.domain.audit_math import AuditOutcome
from src.domain.embedding_state import EmbeddingState
from src.domain.geo_box import BOX_MESSAGE, POLYGON_MESSAGE, parse_geo_box, parse_geo_polygon
from src.domain.viz_box import VIZ_BOX_MESSAGE, parse_viz_box
from src.services.curation.item_filter import QUERY_IDS_MAX, ItemFilter, Origin, ReviewStatus


TakenDate = Annotated[
    dt.date | None,
    Query(
        description=(
            'Capture date (ISO 8601 day), inclusive. Items are matched on the UTC capture time '
            'when the offset was known and on the wall-clock capture time (read as UTC) when '
            'it was not; items with no capture time never match.'
        )
    ),
]


def _parsed[T](name: str, raw: str | None, parse: Callable[[str], T], message: str) -> T | None:
    """``parse(raw)``; a malformed value is the stock FastAPI 422 with the served ``message``."""
    if raw is None:
        return None
    try:
        return parse(raw)
    except ValueError as exc:
        error = {'type': 'value_error', 'loc': ('query', name), 'msg': message, 'input': raw}
        raise RequestValidationError([error]) from exc


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
    detector_class_name: Annotated[
        list[str] | None,
        Query(
            description=(
                "The detector's own class by NAME, repeatable (OR); same normalization as "
                '`class_name`.'
            )
        ),
    ] = None,
    detector_conf_min: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description='Detector confidence, inclusive lower bound.'),
    ] = None,
    detector_conf_max: Annotated[
        float | None,
        Query(
            ge=0.0,
            le=1.0,
            description=(
                'Detector confidence, EXCLUSIVE upper bound (so adjacent histogram bins '
                'never overlap).'
            ),
        ),
    ] = None,
    review_tab: Annotated[
        str | None,
        Query(
            description=(
                'Only items this review queue or filter preset would show: a queue id from '
                '`GET /review/tabs` (`to_label`, `check_machine`, `disagreements`, '
                '`new_class_proposals`, `audit`) or an older preset such as `outliers` or '
                '`detector_disagreements`; `regions` is not accepted.'
            )
        ),
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
    import_id: Annotated[
        str | None,
        Query(description='Items a dataset import labeled (`import_ids`) or proposed.'),
    ] = None,
    audit_outcome: Annotated[
        list[AuditOutcome] | None,
        Query(
            description='Audited crops a person labelled, by who was right (agree, '
            'detector_wrong, vlm_wrong, both_wrong), repeatable (OR): the accuracy audit '
            "report's counts, listed."
        ),
    ] = None,
    taken_from: TakenDate = None,
    taken_to: TakenDate = None,
    cluster_uid: Annotated[
        list[uuid.UUID] | None,
        Query(
            max_length=QUERY_IDS_MAX,
            description='Members of these clusters of the served partition, repeatable (OR; at '
            'most 200).',
        ),
    ] = None,
    geo_bbox: Annotated[
        str | None,
        Query(
            description='Only items on images inside this area: west,south,east,north in '
            'degrees (west > east crosses the antimeridian). Rounded outward under a coarse '
            'location policy; matches nothing under `drop`.'
        ),
    ] = None,
    geo_polygon: Annotated[
        str | None,
        Query(
            description='Only items on images inside this ring: lon,lat;lon,lat;... with 3 to '
            '64 vertices (closed for you). Rounded to the coarse grid under a coarse location '
            'policy; matches nothing under `drop`.'
        ),
    ] = None,
    has_location: Annotated[
        bool | None,
        Query(description='true = items on images with a served position; false = without one.'),
    ] = None,
    near_dup_group: Annotated[
        uuid.UUID | None,
        Query(description='Items in this near-duplicate group (or on images in it).'),
    ] = None,
    viz_box: Annotated[
        str | None,
        Query(
            description='Only items whose embedding-scatter point (`viz_x`, `viz_y`) is inside '
            'this box: x1,y1,x2,y2 with x1 <= x2 and y1 <= y2.'
        ),
    ] = None,
    image_id: Annotated[
        list[str] | None,
        Query(
            max_length=QUERY_IDS_MAX,
            description='Only items on these images, repeatable (at most 200); the map opens '
            'the list on the images of one cell this way.',
        ),
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
        detector_class_names=detector_class_name or [],
        detector_conf_min=detector_conf_min,
        detector_conf_max=detector_conf_max,
        review_tab=review_tab,
        origin=origin or [],
        embedding_state=embedding_state or [],
        review_status=review_status or [],
        audit_outcome=audit_outcome or [],
        import_id=import_id,
        taken_from=taken_from,
        taken_to=taken_to,
        image_ids=image_id or [],
        cluster_uids=cluster_uid or [],
        geo_bbox=_parsed('geo_bbox', geo_bbox, parse_geo_box, BOX_MESSAGE),
        geo_polygon=_parsed('geo_polygon', geo_polygon, parse_geo_polygon, POLYGON_MESSAGE),
        has_location=has_location,
        near_dup_group=near_dup_group,
        viz_box=_parsed('viz_box', viz_box, parse_viz_box, VIZ_BOX_MESSAGE),
    )


ItemFilterQuery = Annotated[ItemFilter, Depends(item_filter_query)]

__all__ = ['ItemFilterQuery', 'item_filter_query']
