"""S-2: heartbeat-based worker healthchecks (src/services/curation/worker_liveness.py)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.services.curation import worker_liveness
from src.services.curation.worker_liveness import check_heartbeat, write_heartbeat


@pytest.fixture
def heartbeat_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the in-process module at a fresh tmp dir.

    ``HEARTBEAT_DIR`` is resolved once at import time (by design — the
    healthcheck always runs as a fresh subprocess in the same container,
    so env-var re-read there isn't needed). In-process tests patch the
    module attribute directly; the CLI subprocess test below still sets
    the env var since that spawns a brand new interpreter.
    """
    hb_dir = tmp_path / 'hb'
    monkeypatch.setenv('OP_HEARTBEAT_DIR', str(hb_dir))
    monkeypatch.setattr(worker_liveness, 'HEARTBEAT_DIR', hb_dir)
    return hb_dir


def _run_cli(name: str, hb_dir: Path, max_age: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            '-m',
            'src.services.curation.worker_liveness',
            'check',
            name,
            '--max-age',
            str(max_age),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={'OP_HEARTBEAT_DIR': str(hb_dir), 'PATH': '/usr/bin:/bin'},
        capture_output=True,
        text=True,
        check=False,
    )


def test_missing_heartbeat_file_is_unhealthy(heartbeat_dir: Path) -> None:
    healthy, reason = check_heartbeat('detection_worker', max_age_s=120.0)
    assert healthy is False
    assert 'missing' in reason


def test_fresh_heartbeat_is_healthy(heartbeat_dir: Path) -> None:
    write_heartbeat('detection_worker', {'producer': True, 'writer': True})
    healthy, reason = check_heartbeat('detection_worker', max_age_s=120.0)
    assert healthy is True, reason


def test_stale_heartbeat_is_unhealthy(heartbeat_dir: Path) -> None:
    path = heartbeat_dir / 'detection_worker.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'ts': time.time() - 1000, 'pid': 1, 'tasks': {}}))
    healthy, reason = check_heartbeat('detection_worker', max_age_s=120.0)
    assert healthy is False
    assert 'stale' in reason


def test_dead_task_is_unhealthy_even_if_fresh(heartbeat_dir: Path) -> None:
    write_heartbeat('detection_worker', {'producer': True, 'writer': False})
    healthy, reason = check_heartbeat('detection_worker', max_age_s=120.0)
    assert healthy is False
    assert 'writer' in reason


def test_cli_exit_codes(heartbeat_dir: Path) -> None:
    missing = _run_cli('vlm_worker', heartbeat_dir)
    assert missing.returncode == 1
    assert 'missing' in missing.stdout

    write_heartbeat('vlm_worker', {'producer': True})
    healthy = _run_cli('vlm_worker', heartbeat_dir)
    assert healthy.returncode == 0
    assert 'ok' in healthy.stdout

    write_heartbeat('vlm_worker', {'producer': False})
    unhealthy = _run_cli('vlm_worker', heartbeat_dir)
    assert unhealthy.returncode == 1


def test_cli_by_path_imports_nothing_heavy(heartbeat_dir: Path) -> None:
    """The compose healthcheck runs the module by file path under a 5 s
    timeout; importing ``src.services`` (Triton client, ultralytics) there
    takes ~6 s and turned every worker unhealthy. Keep the CLI stdlib-only."""
    write_heartbeat('detection_worker', {'producer': True})
    script = Path(__file__).resolve().parents[2] / 'src/services/curation/worker_liveness.py'
    proc = subprocess.run(
        [sys.executable, '-X', 'importtime', str(script), 'check', 'detection_worker'],
        env={'OP_HEARTBEAT_DIR': str(heartbeat_dir), 'PATH': '/usr/bin:/bin'},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    heavy = [m for m in ('src.', 'ultralytics', 'numpy', 'tritonclient') if m in proc.stderr]
    assert not heavy, heavy
