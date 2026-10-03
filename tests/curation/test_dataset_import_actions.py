from __future__ import annotations

from src.services.curation.dataset_import.actions import import_actions


def test_running_import_can_only_cancel() -> None:
    a = import_actions({'status': 'running'}, other_live_id=None)
    assert (a.can_cancel.allowed, a.can_resume.allowed, a.can_undo.allowed) == (True, False, False)


def test_another_live_import_blocks_resume_and_undo_with_its_id() -> None:
    a = import_actions({'status': 'interrupted'}, other_live_id='imp9')
    assert not a.can_resume.allowed
    assert 'imp9' in (a.can_resume.reason or '')
    assert 'imp9' in (a.can_undo.reason or '')


def test_an_undo_run_never_resumes() -> None:
    a = import_actions({'status': 'failed', 'mode': 'undo'}, other_live_id=None)
    assert not a.can_resume.allowed
    assert a.can_undo.allowed
