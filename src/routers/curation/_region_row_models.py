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

    Key a row by ``row_key`` (equivalently ``(crop_id, region_box_id)``). The route-specific keys are
    present only on that route's rows.
    """

    region_box_id: str | None = Field(
        None,
        description=(
            'The box this row is about; null for an item-level row (an item '
            'selected without a box predicate).'
        ),
    )
    row_key: str = Field(
        description=(
            'Non-empty, unique within a page and across pages of one query: '
            '`<crop_id>#<region_box_id>`, or `<crop_id>#item` for an item-level row. '
            'Key a list on this, not on region_box_id (null on item-level rows).'
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


class RegionRowBase(BaseModel):
    items: list[RegionRow]
    total: int = Field(
        description='Items matching. suspected_false_positives pages rows directly, so there '
        'total == total_rows.'
    )
    total_rows: int = Field(description='Rows matching (boxes when the request selects boxes).')
    rows_truncated: bool = Field(
        default=False,
        description='True when an item on this page matched more boxes than the '
        'index reports per item (index.max_inner_result_window), so some of its '
        'rows are missing; total_rows still counts them.',
    )


class RegionRowPage(RegionRowBase):
    """``GET /regions`` and ``/regions/training_candidates``."""

    next_cursor: str | None = Field(
        description='Pass as `cursor` for the next page of items; null on the last page.'
    )
    limit: int = Field(description='Items per page.')


class RegionFpPage(RegionRowBase):
    """``GET /regions/suspected_false_positives``: rows scored in Python, paged by number."""

    page: int
    page_size: int = Field(description='Rows per page.')
