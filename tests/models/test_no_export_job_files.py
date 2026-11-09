"""5B-3: the file protocol of the export worker is gone: no job directory, no heartbeat file,
no claim rename, no private timeout constant."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config.retired_env import reject_retired_env


ROOT = Path(__file__).resolve().parents[2]
SCANNED = ('src', 'export', 'scripts', 'docker')
FORBIDDEN = (
    '/jobs/model_export',
    'worker.heartbeat',
    'reconcile_claimed',
    'EXPORT_TIMEOUT_S',
    'export_task_dir',
)


def _files() -> list[Path]:
    return [
        p
        for root in SCANNED
        for p in (ROOT / root).rglob('*')
        if p.is_file() and p.suffix in {'.py', '.sh', '.yml', '.yaml'}
    ]


def test_nothing_in_the_tree_names_the_export_job_files() -> None:
    pattern = re.compile('|'.join(re.escape(f) for f in FORBIDDEN))
    hits = [
        f'{p.relative_to(ROOT)}:{n}'
        for p in _files()
        if p.relative_to(ROOT).as_posix() != 'src/config/retired_env.py'
        for n, line in enumerate(p.read_text(errors='replace').splitlines(), 1)
        if pattern.search(line) and 'src/services/upgrade' not in p.as_posix()
    ]
    assert hits == []


def test_the_export_worker_has_no_watch_loop() -> None:
    source = (ROOT / 'export' / 'export_worker.py').read_text()
    for gone in (
        'def main',
        'claim_next',
        'signal.signal',
        'import threading',
        'import subprocess',
    ):
        assert gone not in source


def test_the_old_task_dir_variable_is_rejected_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EXPORT_TASK_DIR', '/jobs/model_export')
    with pytest.raises(RuntimeError, match='EXPORT_TASK_DIR'):
        reject_retired_env()
