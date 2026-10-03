"""What an import job lets an operator do right now, decided once.

The route guards (``check_undoable`` / ``check_resumable`` / cancel) and the
served ``actions`` block on the job both read these blockers, so the buttons
a client shows and the 409s the routes raise can never disagree.
Each blocker returns the reason an action is unavailable, or ``None``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.services.curation.dataset_import.store import (
    ACTIVE_STATUSES,
    RESUMABLE_STATUSES,
    UNDOABLE_STATUSES,
)


class ImportAction(BaseModel):
    allowed: bool
    reason: str | None = None


class ImportActions(BaseModel):
    can_cancel: ImportAction
    can_resume: ImportAction
    can_undo: ImportAction


def cancel_blocker(state: dict[str, Any]) -> str | None:
    if state.get('status') in ACTIVE_STATUSES:
        return None
    return 'the import is not running'


def resume_blocker(state: dict[str, Any]) -> str | None:
    if state.get('mode') == 'undo':
        return 'an undo run cannot be resumed'
    if state.get('status') not in RESUMABLE_STATUSES:
        return 'only an interrupted, failed or cancelled import resumes'
    return None


def undo_blocker(state: dict[str, Any]) -> str | None:
    if state.get('status') not in UNDOABLE_STATUSES:
        return 'an import that is running cannot be undone'
    return None


def busy_blocker(other_live_id: str | None) -> str | None:
    if other_live_id is None:
        return None
    return f'another import ({other_live_id}) is running in this project'


def _action(*blockers: str | None) -> ImportAction:
    reason = next((b for b in blockers if b), None)
    return ImportAction(allowed=reason is None, reason=reason)


def import_actions(state: dict[str, Any], *, other_live_id: str | None) -> ImportActions:
    """``other_live_id``: a live import of the project other than this one
    (a start, resume and undo each refuse while one exists)."""
    busy = busy_blocker(other_live_id)
    return ImportActions(
        can_cancel=_action(cancel_blocker(state)),
        can_resume=_action(resume_blocker(state), busy),
        can_undo=_action(undo_blocker(state), busy),
    )
