"""``POST /pause`` / ``POST /resume`` -- per-project pipeline pause.

A pause is ``projects.paused_at``. The workers read it from the registry snapshot at the start of
each cycle and skip a paused project while serving the others; the global GPU pause (the
trainer's single-GPU claim) still pauses every project's workers regardless.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.config.curation_accessors import get_curation_config
from src.config.project_context import current_project
from src.routers.curation._common import router
from src.services.projects import pause as pause_service


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
    from src.services.training.gpu_arbiter_policy import read_training_lock

    lock = read_training_lock()
    if lock is None:
        return None
    devices = lock.get('cuda_visible_devices') or ''
    return f'GPU training claim active (cuda_visible_devices={devices!r})'


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
    await pause_service.set_paused(cfg.project_slug, paused=True)
    _publish_pause_event(event_type='project.paused', cfg=cfg)
    return _state(project_paused=True)


@router.post('/resume', response_model=PipelinePauseState)
async def resume_pipeline() -> PipelinePauseState:
    """Clear this project's pause flag. Idempotent -- resuming an
    already-running project is a no-op, not an error."""
    cfg = get_curation_config()
    await pause_service.set_paused(cfg.project_slug, paused=False)
    _publish_pause_event(event_type='project.resumed', cfg=cfg)
    return _state(project_paused=False)


@router.get('/pause', response_model=PipelinePauseState)
async def get_pipeline_pause_state() -> PipelinePauseState:
    return _state(project_paused=current_project().record.paused_at is not None)


__all__ = [
    'PipelinePauseState',
    'get_pipeline_pause_state',
    'pause_pipeline',
    'resume_pipeline',
]
