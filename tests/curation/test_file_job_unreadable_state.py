"""Issue #161: a state file that exists but cannot be read must not look idle.

``FileJob.is_live`` is the busy check behind every job's start guard (promote,
dataset import, combine, bulk ingest, auto label). Reading it as idle would start a second job beside one
that may still be running."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.services.curation.file_job import FileJob


if TYPE_CHECKING:
    from pathlib import Path

ACTIVE = frozenset({'running'})


def test_a_missing_state_is_idle(tmp_path: Path) -> None:
    job = FileJob(tmp_path)
    assert job.is_live(ACTIVE) is False
    assert job.state_problem() is None


def test_a_readable_idle_state_is_idle(tmp_path: Path) -> None:
    job = FileJob(tmp_path)
    job.write({'status': 'completed'})
    assert job.is_live(ACTIVE) is False
    assert job.state_problem() is None


@pytest.mark.parametrize('content', ['{not json', '[1, 2]', '"running"', ''])
def test_an_unreadable_or_non_object_state_counts_as_live(tmp_path: Path, content: str) -> None:
    job = FileJob(tmp_path)
    job.state_file.write_text(content)
    assert job.is_live(ACTIVE) is True
    assert job.state_problem()


def test_a_state_that_cannot_be_opened_counts_as_live(tmp_path: Path) -> None:
    job = FileJob(tmp_path)
    job.state_file.mkdir()  # reading a directory raises OSError (IsADirectoryError)
    assert job.is_live(ACTIVE) is True


def test_read_still_returns_empty_for_status_polls(tmp_path: Path) -> None:
    job = FileJob(tmp_path)
    job.state_file.write_text('{not json')
    assert job.read() == {}
