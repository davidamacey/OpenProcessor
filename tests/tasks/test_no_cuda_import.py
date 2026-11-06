"""Importing the Celery app must not pull in CUDA libraries or open a database engine.

The prefork parent never initializes CUDA; a child that forks after the parent did would
inherit a broken context.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

_PROBE = """
import sys
sys.path.insert(0, '.')
import src.tasks.app
banned = [m for m in ('torch', 'cuml', 'tritonclient', 'pynvml', 'sqlalchemy') if m in sys.modules]
print(','.join(banned))
"""


def test_importing_the_app_leaves_cuda_libraries_and_engines_out() -> None:
    result = subprocess.run(
        [sys.executable, '-I', '-c', _PROBE],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.stdout.strip() == ''
