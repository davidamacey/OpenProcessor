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

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from scripts.curation._project_worker_utils import PIPELINE_PAUSED_FLAG_NAME
from src.config.curation import get_curation_config
from src.routers.curation._common import router


if TYPE_CHECKING:
    from pathlib import Path


class PipelinePauseState(BaseModel):
    project: str
    paused: bool
    # Cropwright BA-P2-4: whether this project's own flag, the global
    # GPU/training claim, or both are holding its workers idle -- e.g.
    # ['project'], ['gpu_training'], ['project', 'gpu_training'], or [].
    paused_by: list[str] = []
    reason: str | None = None


def _global_gpu_training_reason() -> str | None:
    """Best-effort: is the trainer's single-GPU claim (§ gpu_arbiter) live
    right now? Never fatal -- an unreadable lock file just reports no
    global pause, same as no lock at all."""
    from src.services.training.gpu_arbiter import read_training_lock

    lock = read_training_lock()
    if lock is None:
        return None
    devices = lock.get('cuda_visible_devices') or ''
    return f'GPU training claim active (cuda_visible_devices={devices!r})'


def _flag_path() -> Path:
    return get_curation_config().project_state_dir / PIPELINE_PAUSED_FLAG_NAME


def _state(*, project_paused: bool) -> PipelinePauseState:
    cfg = get_curation_config()
    gpu_reason = _global_gpu_training_reason()
    paused_by: list[str] = []
    if project_paused:
        paused_by.append('project')
    if gpu_reason is not None:
        paused_by.append('gpu_training')
    reason = gpu_reason if (gpu_reason is not None and not project_paused) else None
    return PipelinePauseState(
        project=cfg.project_slug,
        paused=project_paused or gpu_reason is not None,
        paused_by=paused_by,
        reason=reason,
    )


def _publish_pause_event(*, event_type: str, cfg: Any) -> None:
    """BA-P2-5: same envelope style as the M5-step-9 project.* events
    (``target`` names the affected slug)."""
    from src.services.curation.event_hub import publish_global_event

    publish_global_event(event_type, target=cfg.project_slug)


@router.post('/pause', response_model=PipelinePauseState)
async def pause_pipeline() -> PipelinePauseState:
    """Skip this project in every worker's next discovery cycle. Idempotent."""
    cfg = get_curation_config()
    path = _flag_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    _publish_pause_event(event_type='project.paused', cfg=cfg)
    return _state(project_paused=True)


@router.post('/resume', response_model=PipelinePauseState)
async def resume_pipeline() -> PipelinePauseState:
    """Clear this project's pause flag. Idempotent -- resuming an
    already-running project is a no-op, not an error."""
    cfg = get_curation_config()
    _flag_path().unlink(missing_ok=True)
    _publish_pause_event(event_type='project.resumed', cfg=cfg)
    return _state(project_paused=False)


@router.get('/pause', response_model=PipelinePauseState)
async def get_pipeline_pause_state() -> PipelinePauseState:
    return _state(project_paused=_flag_path().exists())


__all__ = [
    'PIPELINE_PAUSED_FLAG_NAME',
    'PipelinePauseState',
    'get_pipeline_pause_state',
    'pause_pipeline',
    'resume_pipeline',
]
