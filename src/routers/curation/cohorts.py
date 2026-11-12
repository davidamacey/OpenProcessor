"""Curation router sub-module — the training-cohort catalog."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Query

from src.clients.curation_opensearch.msearch import msearch_checked
from src.config import get_region_fields
from src.config.curation_accessors import items_index
from src.routers.curation._common import OpenSearchDep, RegistryDep, router
from src.routers.curation._crop_params import CropBrowse, resolve_listing
from src.routers.curation.regions import _training_candidate_query, training_candidates
from src.schemas.curation.cohorts import TrainingCohortList, TrainingCohortSummary
from src.services.curation.cohort_summary import UNDECIDED, agg_body, build_summary
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
    query = await _crop_cohort_query(opensearch, params, class_id)
    resp = await opensearch.count(index=items_index(), body={'query': query})
    return int(resp['count'])


async def _crop_cohort_query(
    opensearch: Any, params: dict[str, Any], class_id: int | None
) -> dict[str, Any]:
    """The query behind a crop cohort's rows (``GET /crops`` with its ``params``), for one class
    or, with no ``class_id``, for every item."""
    listing = await resolve_listing(
        opensearch,
        ItemFilter(class_id=class_id, review_tab=params.get('review_tab')),
        CropBrowse(
            label_validated=params.get('label_validated'),
            classifier_conf_lt=params.get('classifier_conf_lt'),
        ),
    )
    return dict(listing.query_clause)


async def cohort_search_query(opensearch: Any, cohort: dict[str, Any]) -> dict[str, Any]:
    """The query that serves a cohort's rows over every class: a crop cohort's listing, or a
    region cohort's ``training_candidates`` mode (the items with a matching box)."""
    params = cohort['params']
    if cohort['row_kind'] != 'region':
        return await _crop_cohort_query(opensearch, params, None)
    query, box_clause, _reason = _training_candidate_query(params['mode'])
    filters = list(query['bool']['filter'])
    if box_clause is not None:
        filters.append({'nested': {'path': get_region_fields().boxes, 'query': box_clause}})
    return {'bool': {'filter': filters, 'must_not': query['bool']['must_not']}}


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


@router.get('/training_cohorts/summary', response_model=TrainingCohortSummary)
async def training_cohort_summary(
    opensearch: OpenSearchDep, registry: RegistryDep
) -> dict[str, Any]:
    """Every class x every cohort, counted in one request: `{cohorts, rows}`.

    `cohorts` are the catalog's (`GET /training_cohorts`) with a `total`; `rows` has one row per
    active class (by id) with its count in each cohort, then one per detector label that is on
    items without a class and that the registry has no class for (`kind: detector`). The
    cohort *Detector proposals not decided* (the `to_label` queue: the item has no class yet) is
    listed only while such items exist, and is counted by the detector's own label, so a label
    the registry lacks is visible. The counts come from one search per cohort (never per class),
    sent as one multi-search.
    """
    cohorts = cohort_catalog()
    bodies = [agg_body(await cohort_search_query(opensearch, c), 'class_id') for c in cohorts]
    undecided = await cohort_search_query(
        opensearch, {'params': UNDECIDED.params, 'row_kind': UNDECIDED.row_kind}
    )
    bodies.append(agg_body(undecided, 'detector_class_name'))
    responses = await msearch_checked(opensearch, items_index(), bodies)
    return build_summary(
        cohorts,
        {c['id']: responses[i] for i, c in enumerate(cohorts)},
        responses[-1],
        (await registry.snapshot()).classes,
    )
