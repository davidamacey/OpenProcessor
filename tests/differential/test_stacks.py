"""The runs against real stacks (marker ``differential``, never part of the offline suite).

Both tests are the sweep's: ``pytest tests/differential/test_stacks.py -m differential``.

* ``test_every_operation_has_cases`` needs no stack: a dry run over every family of every contract file.
* ``test_family_answers_the_same`` sends one family's cases to ``OP_DIFF_REFERENCE_URL`` and
  ``OP_DIFF_CANDIDATE_URL`` and fails on any difference. A missing URL fails the test, it does not skip it.
  ``OP_DIFF_SEED``, ``OP_DIFF_MAX_EXAMPLES``, ``OP_DIFF_METHODS`` and ``OP_DIFF_CHECK_SCHEMA`` (``1``) are
  passed on; the JSON-lines results go to ``OP_DIFF_OUT_DIR`` (default: the test's temporary directory).

Each run is the command line in a fresh interpreter, not an in-process call: Hypothesis draws from the
literals of the local modules a process has loaded, and a pytest process has loaded many.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from . import families as fam


pytestmark = pytest.mark.differential

REPO = fam.REPO_ROOT


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-m', 'tests.differential.runner', *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_every_operation_has_cases(tmp_path: Path) -> None:
    done = run_cli('--dry-run', '--out', str(tmp_path / 'dry.jsonl'))
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize('family', sorted({f.name for f in fam.load_families()}))
def test_family_answers_the_same(family: str, tmp_path: Path) -> None:
    reference, candidate = (
        os.environ.get('OP_DIFF_REFERENCE_URL'),
        os.environ.get('OP_DIFF_CANDIDATE_URL'),
    )
    assert reference, 'OP_DIFF_REFERENCE_URL is not set: a differential run needs both stacks'
    assert candidate, 'OP_DIFF_CANDIDATE_URL is not set: a differential run needs both stacks'
    out_dir = Path(os.environ.get('OP_DIFF_OUT_DIR', tmp_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    args = ['--families', family, '--out', str(out_dir / f'{family}.jsonl')]
    args += ['--seed', os.environ.get('OP_DIFF_SEED', '0')]
    args += ['--max-examples', os.environ.get('OP_DIFF_MAX_EXAMPLES', '8')]
    if os.environ.get('OP_DIFF_METHODS'):
        args += ['--methods', os.environ['OP_DIFF_METHODS']]
    if os.environ.get('OP_DIFF_CHECK_SCHEMA') == '1':
        args.append('--check-schema')
    done = run_cli('--reference', reference, '--candidate', candidate, *args)
    assert done.returncode == 0, done.stdout + done.stderr
