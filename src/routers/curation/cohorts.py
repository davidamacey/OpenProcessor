"""Curation router sub-module — the training-cohort catalog."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Query

from src.config.curation_accessors import items_index
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._crop_params import CropBrowse, resolve_listing
from src.routers.curation.regions import training_candidates
from src.schemas.curation.cohorts import TrainingCohortList
from src.services.curation.item_filter import ItemFilter
from src.services.curation.training_cohorts import cohort_catalog


async def cohort_count(opensearch: Any, cohort: dict[str, Any], class_id: int) -> int:
    """How many rows ``cohort`` holds for ``class_id``, by the query that serves its rows.

    A crop cohort counts through ``GET /crops``'s listing query; a region cohort reads the
    ``total`` of ``GET /regions/training_candidates``."""
    params = cohort['params']
    if cohort['row_kind'] == 'region':
        page = await training_candidates(
            opensearch, None, mode=params['mode'], cursor=None, limit=1, class_id=class_id
        )
        return int(page['total'])
    listing = await resolve_listing(
        opensearch,
        ItemFilter(class_id=class_id, review_tab=params.get('review_tab')),
        CropBrowse(
            label_validated=params.get('label_validated'),
            classifier_conf_lt=params.get('classifier_conf_lt'),
        ),
    )
    resp = await opensearch.count(index=items_index(), body={'query': listing.query_clause})
    return int(resp['count'])


@router.get('/training_cohorts', response_model=TrainingCohortList)
async def training_cohorts(
    opensearch: OpenSearchDep,
    class_id: Annotated[
        int | None,
        Query(description='Fold into every cohort query and count each cohort for this class.'),
    ] = None,
) -> dict[str, Any]:
    """``{cohorts: [{id, label, description, cutoffs, endpoint, params,
    row_kind, count}]}`` — fetch a cohort's rows with ``GET {prefix}{endpoint}``
    and ``params``. ``count`` is the cohort's size for ``class_id`` (null without one).
    Region cohorts appear only with a region profile."""
    cohorts = cohort_catalog(class_id)
    for cohort in cohorts:
        cohort['count'] = (
            None if class_id is None else await cohort_count(opensearch, cohort, class_id)
        )
    return {'cohorts': cohorts}
