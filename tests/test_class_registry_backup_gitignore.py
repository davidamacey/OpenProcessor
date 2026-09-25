"""F-45 (fresh-start E2E findings 2026-09-25): class-registry backup
snapshots must be gitignored.

Every class-registry write (``ClassRegistry._write`` in
``src/clients/curation_opensearch.py``) leaves a timestamped backup
snapshot next to the live file:
``data/class_registry.<%Y%m%dT%H%M%S%fZ>.json``, e.g.
``data/class_registry.20260925T180317679132Z.json``. A prior attempt at
this fix (commit a2abeb9) added only a test asserting the *behavior* but
never added a ``.gitignore`` pattern -- ``git check-ignore`` on a real
backup filename returned nothing, so 15+ untracked files accumulate in
``git status`` on any active deployment. This test actually calls
``git check-ignore`` (subprocess, real git, not a string match against
.gitignore's text) so it fails the same way a human run of
``git status`` would.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_check_ignore(path: str) -> bool:
    """True if `path` is gitignored (exit 0), False if not (exit 1)."""
    result = subprocess.run(
        ['git', 'check-ignore', '--quiet', path],
        check=False,
        cwd=REPO_ROOT,
        timeout=10,
    )
    return result.returncode == 0


def test_a_real_registry_backup_filename_is_gitignored() -> None:
    # The exact format ClassRegistry._write's snapshot_path constructs:
    # f'{self.path.stem}.{ts}.json' with ts = strftime('%Y%m%dT%H%M%S%fZ').
    assert _git_check_ignore('data/class_registry.20260925T180317679132Z.json')


def test_several_backup_timestamps_across_a_session_are_all_ignored() -> None:
    for ts in (
        '20260101T000000000000Z',
        '20260925T235959999999Z',
        '20261231T120000123456Z',
    ):
        path = f'data/class_registry.{ts}.json'
        assert _git_check_ignore(path), f'{path} is not gitignored'


def test_the_live_registry_and_example_are_unaffected() -> None:
    # /data/class_registry.json (the live file) is separately gitignored
    # (its own pattern); data/class_registry.example.json is the
    # committed warehouse example and must NOT match the backup pattern.
    assert _git_check_ignore('data/class_registry.json')
    assert not _git_check_ignore('data/class_registry.example.json')
