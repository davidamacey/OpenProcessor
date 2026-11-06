"""The jobs-beat healthcheck: a fresh celerybeat-schedule* file passes; none or a stale one fails."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from compose._helpers import _services


if TYPE_CHECKING:
    from pathlib import Path


def _probe_source(directory: Path) -> str:
    test = _services()['jobs-beat']['healthcheck']['test']
    assert test[:3] == ['CMD', 'python', '-c']
    return test[3].replace('/run/beat', str(directory))


def _run(directory: Path) -> int:
    return subprocess.run(
        [sys.executable, '-c', _probe_source(directory)], check=False, timeout=30
    ).returncode


def test_an_empty_glob_is_unhealthy(tmp_path: Path) -> None:
    assert _run(tmp_path) == 1


def test_a_fresh_schedule_file_is_healthy(tmp_path: Path) -> None:
    (tmp_path / 'celerybeat-schedule.db').write_text('x')
    assert _run(tmp_path) == 0


def test_a_stale_schedule_file_is_unhealthy(tmp_path: Path) -> None:
    f = tmp_path / 'celerybeat-schedule'
    f.write_text('x')
    old = time.time() - 600
    os.utime(f, (old, old))
    assert _run(tmp_path) == 1


def test_the_newest_of_several_files_decides(tmp_path: Path) -> None:
    old = tmp_path / 'celerybeat-schedule.dat'
    old.write_text('x')
    past = time.time() - 600
    os.utime(old, (past, past))
    (tmp_path / 'celerybeat-schedule.dir').write_text('x')
    assert _run(tmp_path) == 0
