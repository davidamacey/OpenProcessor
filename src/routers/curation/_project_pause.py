"""``POST /pause`` / ``POST /resume`` -- per-project pipeline pause
(projects_plan.md §5.1/§5.2).

The workers (``scripts/curation/_project_worker_utils.py``'s
``is_project_paused``) already read a file sentinel,
``<project_state_dir>/pipeline_paused.flag`` -- that mechanism predates
this route (the plan's ``pipeline.paused`` settings-doc field needs W2's
config store, not merged here). This route is the missing write side:
an operator (or Cropwright) flips the same flag file through the API
instead of reaching into the container's filesystem by hand.

A project's own pause never touches another project: the *global* GPU
pause sentinel (the trainer's single-GPU claim) still pauses every
project's workers regardless of this per-project flag.
"""

from __future__ import annotations

from pydantic import BaseModel

from scripts.curation._project_worker_utils import PIPELINE_PAUSED_FLAG_NAME
from src.config.curation import get_curation_config
from src.routers.curation._common import router


class PipelinePauseState(BaseModel):
    project: str
    paused: bool


@router.post('/pause', response_model=PipelinePauseState)
async def pause_pipeline() -> PipelinePauseState:
    """Skip this project in every worker's next discovery cycle. Idempotent."""
    cfg = get_curation_config()
    path = cfg.project_state_dir / PIPELINE_PAUSED_FLAG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return PipelinePauseState(project=cfg.project_slug, paused=True)


@router.post('/resume', response_model=PipelinePauseState)
async def resume_pipeline() -> PipelinePauseState:
    """Clear this project's pause flag. Idempotent -- resuming an
    already-running project is a no-op, not an error."""
    cfg = get_curation_config()
    path = cfg.project_state_dir / PIPELINE_PAUSED_FLAG_NAME
    path.unlink(missing_ok=True)
    return PipelinePauseState(project=cfg.project_slug, paused=False)


@router.get('/pause', response_model=PipelinePauseState)
async def get_pipeline_pause_state() -> PipelinePauseState:
    cfg = get_curation_config()
    path = cfg.project_state_dir / PIPELINE_PAUSED_FLAG_NAME
    return PipelinePauseState(project=cfg.project_slug, paused=path.exists())


__all__ = [
    'PIPELINE_PAUSED_FLAG_NAME',
    'PipelinePauseState',
    'get_pipeline_pause_state',
    'pause_pipeline',
    'resume_pipeline',
]
