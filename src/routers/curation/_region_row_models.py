"""Wire models for the region list routes: rows and their page envelope.

Documentation/OpenAPI models only -- handlers return
:func:`~src.services.curation.region_rows.search_region_rows` output
directly. Leaf module.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.routers.curation._item_models import ItemDoc


class RegionRow(ItemDoc):
    """A full wire item plus the box the row is about.

    Key a row by ``(crop_id, region_box_id)``. The route-specific keys are
    present only on that route's rows.
    """

    region_box_id: str | None = Field(
        None,
        description=(
            'The box this row is about; null for an item-level row (an item '
            'selected without a box predicate).'
        ),
    )
    selection_reason: str | None = Field(
        None, description='training_candidates: why the row is in the cohort.'
    )
    suspected_fp_distance: float | None = Field(
        None, description='suspected_false_positives: distance to the nearest FP sub-centroid.'
    )
    nearest_fp_subid: str | None = Field(
        None, description='suspected_false_positives: the nearest FP sub-type id.'
    )


class RegionRowPage(BaseModel):
    """``GET /regions``, ``/regions/training_candidates`` and
    ``/regions/suspected_false_positives``."""

    items: list[RegionRow]
    total: int = Field(
        description='Items matching (page math: hasMore = page * page_size < total). '
        'suspected_false_positives pages rows directly, so there total == total_rows.'
    )
    total_rows: int = Field(description='Rows matching (boxes when the request selects boxes).')
    page: int
    page_size: int = Field(description='Items per page (suspected_false_positives: rows).')
