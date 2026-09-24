"""S-1: the vlm worker's lazy `from src...` import must resolve.

The container command used to be `python scripts/curation/vlm_worker.py
--continuous`, which puts `scripts/curation` (not the repo root) on
`sys.path[0]`. The producer's first-ever iteration builds the pending
query, which does a lazy `from src.services.curation.ingest_class_sources
import classifier_class_sources` -- that raised `ModuleNotFoundError: No
module named 'src'` in production, and the only handler around the
producer loop catches `httpx.HTTPError`, so the worker silently never
did any work with a passing pgrep healthcheck.

Fix: the compose command now runs `python -m scripts.curation.vlm_worker`,
which puts the repo root (cwd, `/app` in the container) on `sys.path[0]`.
This test asserts the module is importable and `_build_pending_query`
resolves when invoked the way `-m` actually runs it -- as a script whose
`sys.path[0]` is the *current working directory*, not the script's own
directory.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_vlm_worker_help_via_module_invocation() -> None:
    """`-m scripts.curation.vlm_worker --help` must not raise ModuleNotFoundError."""
    result = subprocess.run(
        [sys.executable, '-m', 'scripts.curation.vlm_worker', '--help'],
        check=False,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert 'ModuleNotFoundError' not in result.stderr


def test_build_pending_query_resolves_src_import_under_module_invocation() -> None:
    """Reproduces the exact failure: import as a bare script vs. `-m`.

    Run as a bare script (`python scripts/curation/vlm_worker.py`), this
    subprocess call fails with `ModuleNotFoundError: No module named
    'src'` because `sys.path[0]` is `scripts/curation`, not the repo
    root. Run with `-m`, it succeeds. This test pins the `-m` behavior.
    """
    code = (
        'from scripts.curation.vlm_worker import _build_pending_query;'
        'q = _build_pending_query(0.80);'
        'assert isinstance(q, dict) and "bool" in q, q'
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        check=False,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert 'ModuleNotFoundError' not in result.stderr


def test_bare_script_invocation_reproduces_the_original_bug() -> None:
    """Documents the bug directly: with `scripts/curation` (not the repo
    root) as `sys.path[0]` -- exactly what bare `python
    scripts/curation/vlm_worker.py` produces -- the lazy `from
    src.services...` import inside `_build_pending_query` fails with
    `ModuleNotFoundError: No module named 'src'`.
    """
    code = (
        'import sys, importlib, importlib.util;'
        'sys.path[0] = sys.argv[1];'
        "sys.modules.pop('scripts', None);"
        "spec = importlib.util.spec_from_file_location('vlm_worker', sys.argv[2]);"
        'mod = importlib.util.module_from_spec(spec);'
        'spec.loader.exec_module(mod);'
        'mod._build_pending_query(0.80)'
    )
    script_dir = str(REPO_ROOT / 'scripts' / 'curation')
    script_path = str(REPO_ROOT / 'scripts' / 'curation' / 'vlm_worker.py')
    result = subprocess.run(
        [sys.executable, '-c', code, script_dir, script_path],
        check=False,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert 'ModuleNotFoundError' in result.stderr
    assert "'src'" in result.stderr
