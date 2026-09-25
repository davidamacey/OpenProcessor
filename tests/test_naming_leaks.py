"""Tests for ``scripts/codegen/check_naming_leaks.py`` (naming sweep W8).

Two kinds of coverage:

1. The real repo, scanned with the real allowlist, must currently pass —
   this is the actual pre-commit hook behavior, run in-process via
   :func:`check_naming_leaks.main` (subprocess would just re-run the same
   check; calling ``main()`` directly is equivalent and faster).
2. The allowlist *mechanics* (whole-path exemption, directory-prefix
   exemption, ``path:regex`` line-level exemption, and the "still fails on
   an unrelated hit in an otherwise-allowlisted file" case) are exercised
   against a synthetic temp git repo via :func:`check_naming_leaks.run_scans`
   directly, so they don't depend on the real repo's current content.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'codegen'))
import check_naming_leaks as leaks


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_repo_passes_the_guard() -> None:
    """The actual hook, run against the actual repo: exit 0."""
    exit_code = leaks.main()
    assert exit_code == 0


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True)
    return repo


def _write_and_track(repo: Path, rel_path: str, content: str) -> None:
    path = repo / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    subprocess.run(['git', 'add', rel_path], cwd=repo, check=True)


def _allowlist(repo: Path, content: str) -> Path:
    path = repo / 'allowlist.txt'
    path.write_text(content)
    return path


def test_scan_a_catches_a_bare_leak(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_and_track(repo, 'src/thing.py', "NAME = 'legacy_stuff'\n")
    allowlist = _allowlist(repo, '')
    hits, _ = leaks.run_scans(repo, allowlist, scans={'A': leaks.SCAN_A})
    assert any('legacy_stuff' in h for h in hits['A'])


def test_whole_path_allowlist_entry_exempts_every_hit_in_that_file(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_and_track(repo, 'src/thing.py', "NAME = 'legacy_stuff'\nOTHER = 'legacy_other'\n")
    allowlist = _allowlist(repo, 'src/thing.py # deliberate test fixture\n')
    hits, entries = leaks.run_scans(repo, allowlist, scans={'A': leaks.SCAN_A})
    assert hits['A'] == []
    assert entries[0].used is True


def test_directory_prefix_allowlist_entry_exempts_files_under_it(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_and_track(repo, 'examples/thing.py', "NAME = 'legacy_stuff'\n")
    allowlist = _allowlist(repo, 'examples/ # domain examples directory\n')
    hits, entries = leaks.run_scans(repo, allowlist, scans={'A': leaks.SCAN_A})
    assert hits['A'] == []
    assert entries[0].used is True


def test_path_regex_entry_exempts_only_matching_lines(tmp_path: Path) -> None:
    """The core allowlist-mechanics case: a `path:regex` entry allows the
    one line it names, but a DIFFERENT leak in the same file still fails --
    this is what makes the allowlist a ratchet instead of a blanket
    per-file exemption."""
    repo = _init_repo(tmp_path)
    _write_and_track(
        repo,
        'tests/test_thing.py',
        "assert 'legacy_retired' not in x\nNEW_LEAK = 'legacy_new_and_unreviewed'\n",
    )
    allowlist = _allowlist(
        repo, 'tests/test_thing.py:legacy_retired # negative test for a retired name\n'
    )
    hits, entries = leaks.run_scans(repo, allowlist, scans={'A': leaks.SCAN_A})
    assert len(hits['A']) == 1
    assert 'legacy_new_and_unreviewed' in hits['A'][0]
    assert 'legacy_retired' not in hits['A'][0]
    assert entries[0].used is True


def test_unused_allowlist_entry_is_reported(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_and_track(repo, 'src/thing.py', "NAME = 'clean'\n")
    allowlist = _allowlist(repo, 'src/nonexistent_leak.py # stale entry, nothing matches it\n')
    hits, entries = leaks.run_scans(repo, allowlist, scans={'A': leaks.SCAN_A})
    assert hits['A'] == []
    assert entries[0].used is False


def test_excluded_path_is_never_scanned_even_without_an_allowlist_entry(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_and_track(repo, 'docs/design/naming_sweep_plan.md', 'mentions legacy_stuff on purpose\n')
    allowlist = _allowlist(repo, '')
    hits, _ = leaks.run_scans(
        repo,
        allowlist,
        exclude_paths=('docs/design/naming_sweep_plan.md',),
        scans={'A': leaks.SCAN_A},
    )
    assert hits['A'] == []


@pytest.mark.parametrize(
    'raw_line',
    [
        'no_hash_or_reason\n',
        'path.py:regex\n',  # has a colon but still no '#'
    ],
)
def test_malformed_allowlist_entry_raises(tmp_path: Path, raw_line: str) -> None:
    repo = _init_repo(tmp_path)
    allowlist = _allowlist(repo, raw_line)
    with pytest.raises(ValueError, match='reason'):
        leaks.load_allowlist(allowlist)


def test_allowlist_entry_with_empty_reason_raises(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    allowlist = _allowlist(repo, 'src/thing.py #   \n')
    with pytest.raises(ValueError, match='reason'):
        leaks.load_allowlist(allowlist)


def test_missing_allowlist_file_means_no_exemptions(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_and_track(repo, 'src/thing.py', "NAME = 'legacy_stuff'\n")
    hits, entries = leaks.run_scans(
        repo, tmp_path / 'does_not_exist.txt', scans={'A': leaks.SCAN_A}
    )
    assert entries == []
    assert any('legacy_stuff' in h for h in hits['A'])
