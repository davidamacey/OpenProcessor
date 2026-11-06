"""The status an import serves, derived from its job rows and the import row.

An import has a run job (``dataset_imports.job_id``) and, once undone or being undone, an undo
job (``undo_job_id``). The wire keeps the vocabulary clients know: ``queued``, ``running``,
``paused_backpressure``, ``completed``, ``completed_with_errors``, ``failed``, ``cancelled``,
``interrupted``, ``undoing``, ``undone``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.db.repos.jobs import JobRow
    from src.services.curation.dataset_import.rows import ImportRecord

ACTIVE_STATUSES = frozenset({'queued', 'running', 'paused_backpressure', 'undoing'})
COMPLETED_STATUSES = frozenset({'completed', 'completed_with_errors'})
RESUMABLE_STATUSES = frozenset({'interrupted', 'failed', 'cancelled'})
UNDOABLE_STATUSES = frozenset(
    {'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted', 'undone'}
)
BACKPRESSURE_STAGE = 'paused_backpressure'

_JOB_ACTIVE = frozenset({'queued', 'running', 'cancelling', 'paused'})
_RUNNING = frozenset({'running', 'cancelling', 'paused'})


def _from_run(run: JobRow | None) -> str:
    if run is None:
        return 'failed'
    if run.status in _RUNNING:
        stage = (run.progress or {}).get('stage')
        return BACKPRESSURE_STAGE if stage == BACKPRESSURE_STAGE else 'running'
    return run.status


def import_status(record: ImportRecord, run: JobRow | None, undo: JobRow | None) -> str:
    """The wire status, and (via :func:`import_mode`) which job it describes."""
    if undo is not None and undo.status in _JOB_ACTIVE:
        return 'undoing'
    if record.undone_at is not None:
        return 'undone'
    if undo is not None and undo.status != 'completed' and _undo_is_latest(run, undo):
        return 'failed'
    return _from_run(run)


def _undo_is_latest(run: JobRow | None, undo: JobRow) -> bool:
    return run is None or undo.created_at >= run.created_at


def import_mode(record: ImportRecord, run: JobRow | None, undo: JobRow | None) -> str:
    if undo is not None and (
        undo.status in _JOB_ACTIVE
        or record.undone_at is not None
        or (undo.status != 'completed' and _undo_is_latest(run, undo))
    ):
        return 'undo'
    return 'import'


def state_of(record: ImportRecord, run: JobRow | None, undo: JobRow | None) -> dict[str, Any]:
    """The ``{status, mode}`` pair the action blockers decide on."""
    return {'status': import_status(record, run, undo), 'mode': import_mode(record, run, undo)}
