"""``/metrics`` aggregates every API worker when PROMETHEUS_MULTIPROC_DIR is set.

Each fake worker is a separate process (prometheus_client keys its value
files by pid), so this proves the same aggregation the 32 uvicorn workers
rely on, not just a single registry.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

_WORKER = (
    'from src.core.metrics import HTTP_REQUEST_DURATION_SECONDS as H\n'
    'for _ in range({n}):\n'
    "    H.labels('GET', '/x', '200').observe(0.01)\n"
)


def _run(code: str, env_dir: Path | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != 'PROMETHEUS_MULTIPROC_DIR'}
    if env_dir is not None:
        env['PROMETHEUS_MULTIPROC_DIR'] = str(env_dir)
    env['PYTHONPATH'] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, '-c', code],
        env=env,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _render() -> str:
    return 'from src.core.metrics import render_metrics\nprint(render_metrics()[0].decode())\n'


def _count(body: str) -> float:
    m = re.search(r'^http_request_duration_seconds_count\{[^}]*\} (\S+)$', body, re.MULTILINE)
    assert m, body
    return float(m.group(1))


def test_two_workers_are_summed_in_one_scrape(tmp_path: Path) -> None:
    _run(_WORKER.format(n=3), tmp_path)
    _run(_WORKER.format(n=4), tmp_path)
    assert len(list(tmp_path.glob('histogram_*.db'))) == 2
    assert _count(_run(_render(), tmp_path)) == 7


def test_without_the_variable_a_scrape_sees_only_its_own_process(tmp_path: Path) -> None:
    _run(_WORKER.format(n=3), None)
    with pytest.raises(AssertionError):
        _count(_run(_render(), None))
