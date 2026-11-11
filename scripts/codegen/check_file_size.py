#!/usr/bin/env python3
"""Pre-commit guard: file size is a guide to separation of concerns, not a hard cap.

Scope, by kind of file:

* running application code (``src``, ``models``, ``docker`` and ``export`` Python,
  ``frontend/src`` ``.ts .svelte .js`` except ``*.test.ts``): warn from 700, hard cap 1000;
* test code (``tests``, ``frontend/e2e``, ``*.test.ts``): warn from 700, hard cap 1500;
* out of scope, naturally long: ``scripts/``, every ``.sh``, the root CLI programs,
  ``frontend/scripts/``, vendored bundles (``src/static/``).

Every ``CLAUDE.md`` has its own cap (400 lines).

Files of 700 to 1000 lines are listed as a warning and never fail: split them by
concern in the change that touches them, never by collapsing lines. Above the hard cap the
hook fails unless the file has a reviewed entry in ``baselines/file_size.txt``
(``path<TAB>lines<TAB>cohesive: <one-line reason>``, a long file that is still one
concern); the baseline can only shrink (see ``_baseline.py``), and an entry whose
reason does not start with ``cohesive:`` is an error.

The hook runs over the whole tracked tree (``pass_filenames: false``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / 'scripts' / 'codegen' / 'baselines' / 'file_size.txt'

APP_DIRS = ('src/', 'models/', 'docker/', 'export/')
APP_SUFFIXES = ('.py',)
FRONTEND_APP_DIR = 'frontend/src/'
FRONTEND_APP_SUFFIXES = ('.ts', '.svelte', '.js')
TEST_DIRS = ('tests/', 'frontend/e2e/')
TEST_SUFFIXES = ('.py', '.ts', '.js', '.mjs')
VENDORED_PREFIXES = ('src/static/',)
CLAUDE_MD_MAX = 400
APP_MAX = 1000
TEST_MAX = 1500


def tracked_files(repo_root: Path) -> list[str]:
    out = subprocess.run(  # nosec B603 B607  fixed argv, git from PATH
        ['git', 'ls-files', '-z'], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split('\0') if p]


def cap_for(path: str) -> int | None:
    """The hard line cap that applies to ``path``, or ``None`` when the file is out of scope."""
    if path.rsplit('/', 1)[-1] == 'CLAUDE.md':
        return CLAUDE_MD_MAX
    if path.startswith(VENDORED_PREFIXES):
        return None
    if path.endswith('.test.ts') and path.startswith('frontend/'):
        return TEST_MAX
    if path.startswith(TEST_DIRS) and path.endswith(TEST_SUFFIXES):
        return TEST_MAX
    if path.startswith(APP_DIRS) and path.endswith(APP_SUFFIXES):
        return APP_MAX
    if path.startswith(FRONTEND_APP_DIR) and path.endswith(FRONTEND_APP_SUFFIXES):
        return APP_MAX
    return None


def line_count(path: Path) -> int:
    try:
        with path.open('rb') as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def scan(
    repo_root: Path, *, warn_lines: int, files: list[str] | None = None
) -> tuple[dict[str, int], dict[str, int]]:
    violations: dict[str, int] = {}
    warnings: dict[str, int] = {}
    for rel in files if files is not None else tracked_files(repo_root):
        cap = cap_for(rel)
        if cap is None:
            continue
        n = line_count(repo_root / rel)
        if n > cap:
            violations[rel] = n
        elif cap != CLAUDE_MD_MAX and n >= warn_lines:
            warnings[rel] = n
    return violations, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warn', type=int, default=700, help='Warn band start (never fails).')
    parser.add_argument('--baseline', type=Path, default=BASELINE_PATH)
    parser.add_argument('--tighten', action='store_true', help='Lower baseline values, drop stale.')
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    violations, warnings = scan(args.root, warn_lines=args.warn)
    baseline = _baseline.load(args.baseline)
    errors = _baseline.compare(violations, baseline, unit='lines')
    errors += [
        f'{key}: baseline reason must start with "cohesive:" (a reviewed one-concern file)'
        for key, entry in sorted(baseline.items())
        if not entry.reason.startswith('cohesive:')
    ]
    if args.tighten:
        changed = _baseline.tighten(args.baseline, violations, baseline)
        sys.stderr.write(f'tightened {changed} baseline line(s)\n')
        errors = [e for e in errors if 'new violation' in e or 'grew past' in e]

    if warnings:
        sys.stderr.write(
            f'warning: {len(warnings)} file(s) are in the {args.warn}+ line band; '
            'consider splitting them by concern when you touch them:\n'
        )
        for rel, n in sorted(warnings.items()):
            sys.stderr.write(f'  {rel}: {n}\n')
    if errors:
        sys.stderr.write(
            f'ERROR: file size ratchet failed (hard caps: app {APP_MAX}, tests {TEST_MAX}, CLAUDE.md {CLAUDE_MD_MAX}).\n'
            'Split the file by concern, or add a reviewed `cohesive:` baseline entry; never collapse lines to fit.\n'
        )
        for e in errors:
            sys.stderr.write(f'  {e}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
