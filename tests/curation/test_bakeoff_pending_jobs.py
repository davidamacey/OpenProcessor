"""The evaluator's watcher finds a queued job in every project's own job
directory (the API writes ``<state>/projects/<slug>/bakeoff_jobs``), not only
in the shared watch directory."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from scripts.curation.bakeoff.bakeoff_runner import _pending_job_files


if TYPE_CHECKING:
    from pathlib import Path


def _queue(path: Path, mtime: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}')
    os.utime(path, (mtime, mtime))
    return path


def test_every_projects_queue_is_found_oldest_first(tmp_path: Path) -> None:
    watch = tmp_path / 'bakeoff_jobs'
    shared = _queue(watch / 'shared.job.json', 300)
    alpha = _queue(tmp_path / 'projects' / 'alpha' / 'bakeoff_jobs' / 'a.job.json', 100)
    default = _queue(tmp_path / 'projects' / 'default' / 'bakeoff_jobs' / 'd.job.json', 200)
    _queue(tmp_path / 'projects' / 'alpha' / 'bakeoff_jobs' / 'done' / 'old.job.json', 50)

    assert _pending_job_files(watch) == [alpha, default, shared]
