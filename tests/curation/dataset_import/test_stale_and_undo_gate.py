"""What an operator may do with an import right now, decided from its status.

An interrupted run (the reaper's state for a worker that died) is resumable and undoable; a
running one is neither; an undo, like a start, refuses while another import of the project is
live. The gates read one decision (``control.check_*`` and the served ``actions``), so a
button and the 409 behind it cannot disagree."""

from __future__ import annotations

import datetime as dt
import uuid

import pytest

from src.services.curation.dataset_import import control, runner
from src.services.curation.dataset_import.actions import import_actions
from src.services.curation.dataset_import.rows import ImportRecord


def _view(status: str, *, mode: str = 'import', other: str | None = None) -> control.ImportView:
    record = ImportRecord(
        id=uuid.uuid4(),
        job_id=None,
        undo_job_id=None,
        format='yolo',
        mapping={},
        request={},
        scan={},
        report={},
        created_at=dt.datetime.now(dt.UTC),
        undone_at=None,
    )
    return control.ImportView(record, None, None, status, mode, other)


@pytest.mark.parametrize('status', ['interrupted', 'failed', 'cancelled'])
def test_an_interrupted_failed_or_cancelled_import_is_resumable_and_undoable(status) -> None:
    control.check_resumable(_view(status))
    control.check_undoable(_view(status))


@pytest.mark.parametrize('status', ['queued', 'running', 'paused_backpressure', 'undoing'])
def test_a_live_import_is_neither_resumable_nor_undoable(status) -> None:
    with pytest.raises(runner.ImportNotResumableError):
        control.check_resumable(_view(status))
    with pytest.raises(runner.ImportNotUndoableError):
        control.check_undoable(_view(status))


def test_a_completed_import_is_undoable_not_resumable() -> None:
    control.check_undoable(_view('completed'))
    with pytest.raises(runner.ImportNotResumableError):
        control.check_resumable(_view('completed'))


def test_an_undo_run_cannot_be_resumed() -> None:
    with pytest.raises(runner.ImportNotResumableError):
        control.check_resumable(_view('failed', mode='undo'))


def test_undo_and_resume_refuse_while_another_import_is_live() -> None:
    other = str(uuid.uuid4())
    with pytest.raises(runner.ImportBusyError) as busy:
        control.check_undoable(_view('completed', other=other))
    assert busy.value.import_id == other
    with pytest.raises(runner.ImportBusyError):
        control.check_resumable(_view('interrupted', other=other))


def test_the_served_actions_match_the_gates() -> None:
    interrupted = import_actions(_view('interrupted').state, other_live_id=None)
    assert (interrupted.can_resume.allowed, interrupted.can_undo.allowed) == (True, True)
    assert interrupted.can_cancel.allowed is False
    busy = import_actions(_view('completed').state, other_live_id='x')
    assert busy.can_undo.allowed is False
    assert 'another import' in (busy.can_undo.reason or '')
    undoing = import_actions(_view('undoing', mode='undo').state, other_live_id=None)
    assert undoing.can_cancel.allowed is False


def test_undo_keeps_a_box_whose_ledger_row_has_no_bbox() -> None:
    """Ledgers written before standalone items recorded their box have rows
    without ``bbox_norm``: undo must keep that box, not crash."""
    from src.domain.region_boxes import RegionBox
    from src.services.curation.dataset_import.undo import CANDIDATE_IMPORT, _import_boxes

    box = RegionBox(
        box_id='b1',
        bbox_norm=(0.1, 0.1, 0.5, 0.5),
        state='accepted',
        source=CANDIDATE_IMPORT,
        detector_version='imp1',
    )
    mine, edited = _import_boxes([box], 'imp1', [{'crop_id': 'c', 'box_id': 'b1'}])
    assert mine == []
    assert edited == [box]
