"""Each review-filter module must import cleanly as the first import of a process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    'module',
    [
        'src.services.curation.review_request',
        'src.services.curation.review_filter_specs',
        'src.services.curation.review_filters',
        'src.routers.curation.review',
    ],
)
def test_module_imports_alone_in_a_fresh_interpreter(module: str) -> None:
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run(
        [sys.executable, '-c', f'import {module}'],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
    )
    assert done.returncode == 0, done.stderr[-800:]
