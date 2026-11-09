"""OPS-2: a process that leaves takes its live-gauge files out of the multiprocess directory.

``/metrics`` reads every file in ``PROMETHEUS_MULTIPROC_DIR``; 1,000 dead-process file sets cost
about 2.8 s per scrape against 5 ms for none, so the API's shutdown and a Celery child leaving
(for instance under ``--max-tasks-per-child``) call ``mark_process_dead``
(https://prometheus.github.io/client_python/multiprocess/).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.tasks import app as tasks_app
from src.utils.metrics_processes import forget_process


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def make_files(directory: Path, pid: int) -> list[Path]:
    names = (f'gauge_livemostrecent_{pid}.db', f'gauge_livesum_{pid}.db', f'counter_{pid}.db')
    files = [directory / name for name in names]
    for file in files:
        file.write_bytes(b'')
    return files


def test_the_live_gauge_files_of_a_dead_process_go_and_its_counters_stay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('PROMETHEUS_MULTIPROC_DIR', str(tmp_path))
    mine, other = make_files(tmp_path, 4242), make_files(tmp_path, 4343)

    forget_process(4242)

    assert [f.exists() for f in mine] == [False, False, True]  # counters are history, kept
    assert all(f.exists() for f in other)


def test_without_a_multiprocess_directory_nothing_happens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('PROMETHEUS_MULTIPROC_DIR', raising=False)
    forget_process(4242)


def test_a_celery_child_leaving_forgets_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('PROMETHEUS_MULTIPROC_DIR', str(tmp_path))
    mine = make_files(tmp_path, 5151)

    tasks_app._on_worker_process_shutdown(pid=5151, exitcode=0)

    assert [f.exists() for f in mine] == [False, False, True]
