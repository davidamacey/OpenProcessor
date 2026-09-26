from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_bash(
    script: str, *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run a bash snippet, returning the CompletedProcess (never raises)."""
    return subprocess.run(
        ['bash', '-c', script],
        check=False,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def bash():
    return run_bash
