"""What a person may do with an assist loop in each state (pure; the backend decides, the
client renders).

The state is the job's status, with a running job that has a pause requested read as
``pausing``. A resume is offered only when nothing blocks it (the project is paused, or the
VLM is unavailable), and the blocker is named so the client can say why it is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LoopControlState = Literal[
    'queued',
    'running',
    'pausing',
    'paused',
    'cancelling',
    'completed',
    'completed_with_errors',
    'failed',
    'cancelled',
    'interrupted',
]
ResumeBlocker = Literal['project_paused', 'vlm_unavailable']

_LIVE = frozenset({'queued', 'running', 'pausing'})
_STOPPED = frozenset(
    {'paused', 'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted'}
)


@dataclass(frozen=True)
class ControlFlags:
    state: LoopControlState
    can_pause: bool
    can_resume: bool
    can_cancel: bool
    can_edit_budget: bool
    can_rollback: bool
    resume_blocked_reason: ResumeBlocker | None


def control_state(status: str, *, pause_requested: bool) -> LoopControlState:
    if status == 'running' and pause_requested:
        return 'pausing'
    return status  # type: ignore[return-value]


def control_flags(
    status: str, *, pause_requested: bool, blocker: ResumeBlocker | None
) -> ControlFlags:
    state = control_state(status, pause_requested=pause_requested)
    resumable = state in ('paused', 'interrupted')
    return ControlFlags(
        state=state,
        can_pause=state in ('queued', 'running'),
        can_resume=resumable and blocker is None,
        can_cancel=state in _LIVE or resumable,
        can_edit_budget=state in (*_LIVE, 'paused'),
        can_rollback=state in _STOPPED,
        resume_blocked_reason=blocker if resumable else None,
    )


__all__ = ['ControlFlags', 'LoopControlState', 'ResumeBlocker', 'control_flags', 'control_state']
