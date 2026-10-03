"""Curation router sub-module: pause and report the region stage of a project.

``GET /region_stage`` reports whether the stage is paused, how much work is
waiting and what the gate skipped, plus the reprocess request that re-runs the
skipped items. ``POST /region_stage/pause`` and ``POST /region_stage/resume``
flip the stage for this project only (idempotent); like every region route they
409 until a region profile is active. Pausing loses nothing, see
:mod:`~src.services.curation.region_stage_control`.
"""

from __future__ import annotations

from src.routers.curation._common import OpenSearchDep, RegionProfileDep, router
from src.services.curation.region_stage_control import (
    RegionStageState,
    region_stage_state,
    set_region_stage_paused,
)


@router.get('/region_stage', response_model=RegionStageState)
async def get_region_stage(
    opensearch: OpenSearchDep, _profile: RegionProfileDep
) -> RegionStageState:
    return await region_stage_state(opensearch)


@router.post('/region_stage/pause', response_model=RegionStageState)
async def pause_region_stage(
    opensearch: OpenSearchDep, _profile: RegionProfileDep
) -> RegionStageState:
    """Stop spending segmenter time on this project. Items stay
    ``pending_detection``; in-flight items past the segmenter finish."""
    set_region_stage_paused(True)
    return await region_stage_state(opensearch)


@router.post('/region_stage/resume', response_model=RegionStageState)
async def resume_region_stage(
    opensearch: OpenSearchDep, _profile: RegionProfileDep
) -> RegionStageState:
    set_region_stage_paused(False)
    return await region_stage_state(opensearch)
