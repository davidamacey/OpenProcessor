"""Test-order pollution: the keymap isolation tests pass after the clone tests.

The clone tests patch ``src.services.curation.keymap.get_keymap_doc`` by dotted path. A
module that first imports that name while the patch is active keeps the fake for the rest
of the process, so the keymap router served a fake after the clone tests ran. xdist hid
it in CI by spreading the files across workers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ORDERED = [
    'tests/projects/test_clone_settings.py',
    'tests/projects/test_clone_activations.py',
    'tests/curation/cross_project_leak/test_keymap_and_ingest_policy_isolation.py',
]


def test_keymap_isolation_tests_pass_after_the_clone_tests() -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-p', 'no:xdist', '-q', '--no-cov', *ORDERED],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout[-3000:]
