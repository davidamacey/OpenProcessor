"""Pause, resume and report the region stage of one project.

The pause is the file ``region_stage_paused.flag`` in the project's state
directory (the mechanism the pipeline pause already uses, read by the same
workers): the region worker fetches nothing for the project and hands back
items it already holds before their segmenter call, so every item stays
``pending_detection``. Nothing is written to an item and nothing is lost;
resuming just removes the file. Other stages (VLM labelling, ingest) keep
running. Items the crop gate skipped sit in ``no_region_box`` with
``region_gate_skip`` set; the state carries the reprocess request that re-runs
exactly those.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from scripts.curation._project_worker_utils import (
    PIPELINE_PAUSED_FLAG_NAME,
    REGION_STAGE_PAUSED_FLAG_NAME,
)
from src.config import RegionStatus, get_curation_config
from src.config.region_fields import get_region_fields
from src.services.curation.reprocess_models import (
    ReprocessFilter,
    ReprocessRequest,
    ReprocessTargets,
)


if TYPE_CHECKING:
    from pathlib import Path

    from opensearchpy import AsyncOpenSearch


class RegionStageCounts(BaseModel):
    pending_detection: int
    pending_verification: int
    #: Items the gate skipped (``region_gate_skip`` set): never looked at.
    gate_skipped: int


class RegionStageState(BaseModel):
    project: str
    paused: bool
    paused_since: str | None = None
    #: The whole-pipeline pause is on too (it also holds this stage).
    pipeline_paused: bool
    counts: RegionStageCounts
    #: Body for the reprocess route that re-runs the gate-skipped items
    #: (a dry run: set ``dry_run`` false to apply).
    rerun_skipped: ReprocessRequest


def _flag(name: str) -> Path:
    return get_curation_config().project_state_dir / name


def set_region_stage_paused(paused: bool) -> None:
    flag = _flag(REGION_STAGE_PAUSED_FLAG_NAME)
    if paused:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
    else:
        flag.unlink(missing_ok=True)


def rerun_skipped_request() -> ReprocessRequest:
    return ReprocessRequest(
        targets=ReprocessTargets(
            filter=ReprocessFilter(
                region_status=[RegionStatus.NO_REGION_BOX.value], region_gate_skipped=True
            )
        ),
        scopes=['region'],
        dry_run=True,
    )


async def region_stage_state(opensearch: AsyncOpenSearch) -> RegionStageState:
    cfg = get_curation_config()
    F = get_region_fields()

    async def count(query: dict) -> int:
        resp = await opensearch.count(index=cfg.items_index, body={'query': query})
        return int(resp['count'])

    flag = _flag(REGION_STAGE_PAUSED_FLAG_NAME)
    since = datetime.fromtimestamp(flag.stat().st_mtime, UTC).isoformat() if flag.exists() else None
    return RegionStageState(
        project=cfg.project_slug,
        paused=since is not None,
        paused_since=since,
        pipeline_paused=_flag(PIPELINE_PAUSED_FLAG_NAME).exists(),
        counts=RegionStageCounts(
            pending_detection=await count(
                {'term': {F.status: RegionStatus.PENDING_DETECTION.value}}
            ),
            pending_verification=await count(
                {'term': {F.status: RegionStatus.PENDING_VERIFICATION.value}}
            ),
            gate_skipped=await count({'exists': {'field': F.gate_skip}}),
        ),
        rerun_skipped=rerun_skipped_request(),
    )


__all__ = [
    'RegionStageCounts',
    'RegionStageState',
    'region_stage_state',
    'rerun_skipped_request',
    'set_region_stage_paused',
]
