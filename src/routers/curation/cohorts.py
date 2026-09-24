"""Curation router sub-module — the training-cohort catalog."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Query

from src.routers.curation._common import router
from src.services.curation.training_cohorts import cohort_catalog


@router.get('/training_cohorts')
async def training_cohorts(
    class_id: Annotated[int | None, Query(description='Fold into every cohort query.')] = None,
) -> dict[str, Any]:
    """``{cohorts: [{id, label, description, cutoffs, endpoint, params,
    row_kind}]}`` — fetch a cohort's rows with ``GET {prefix}{endpoint}``
    and ``params``. Region cohorts appear only with a region profile."""
    return {'cohorts': cohort_catalog(class_id)}
