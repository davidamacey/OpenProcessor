"""Tests run inside git hooks (pre-push), which export repository-locating variables; none may reach a test."""

from __future__ import annotations

import os
import subprocess


def test_no_repository_local_git_variable_reaches_a_test() -> None:
    # git's own list of the variables it clears when it moves to another repository.
    names = subprocess.run(
        ['git', 'rev-parse', '--local-env-vars'], capture_output=True, text=True, check=True
    ).stdout.split()
    assert names, 'git printed no local variables'
    assert [n for n in names if n in os.environ] == []
