#!/usr/bin/env python3
"""Pre-commit guard: cap the line count of first-party files.

Scope: every tracked ``.py .ts .svelte .js .mjs .sh`` file under ``src``,
``scripts``, ``export``, ``docker``, ``models``, ``tests``, ``frontend/src``,
``frontend/e2e``, ``frontend/scripts``, plus the root shell programs. Vendored
bundles (``src/static/``) are out of scope. Every ``CLAUDE.md`` has its own
cap (400 lines).

A file over its cap must be in ``baselines/file_size.txt`` (transition list,
``path<TAB>lines<TAB>reason``); the baseline can only shrink (see
``_baseline.py``). Files in the 640-700 band are listed as a warning and never
fail: split them by concern in the change that touches them, never by
collapsing lines.

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

SCOPE_DIRS = (
    'src/',
    'scripts/',
    'export/',
    'docker/',
    'models/',
    'tests/',
    'frontend/src/',
    'frontend/e2e/',
    'frontend/scripts/',
)
SCOPE_ROOT_FILES = frozenset({'openprocessor', 'setup-openprocessor.sh'})
SUFFIXES = ('.py', '.ts', '.svelte', '.js', '.mjs', '.sh')
VENDORED_PREFIXES = ('src/static/',)
CLAUDE_MD_MAX = 400


def tracked_files(repo_root: Path) -> list[str]:
    out = subprocess.run(  # nosec B603 B607  fixed argv, git from PATH
        ['git', 'ls-files', '-z'], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split('\0') if p]


def cap_for(path: str, default_max: int) -> int | None:
    """The line cap that applies to ``path``, or ``None`` when the file is out of scope."""
    if path.rsplit('/', 1)[-1] == 'CLAUDE.md':
        return CLAUDE_MD_MAX
    if path.startswith(VENDORED_PREFIXES):
        return None
    if path in SCOPE_ROOT_FILES:
        return default_max
    if path.startswith(SCOPE_DIRS) and path.endswith(SUFFIXES):
        return default_max
    return None


def line_count(path: Path) -> int:
    try:
        with path.open('rb') as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def scan(
    repo_root: Path, *, max_lines: int, warn_lines: int, files: list[str] | None = None
) -> tuple[dict[str, int], dict[str, int]]:
    violations: dict[str, int] = {}
    warnings: dict[str, int] = {}
    for rel in files if files is not None else tracked_files(repo_root):
        cap = cap_for(rel, max_lines)
        if cap is None:
            continue
        n = line_count(repo_root / rel)
        if n > cap:
            violations[rel] = n
        elif cap == max_lines and n >= warn_lines:
            warnings[rel] = n
    return violations, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max', type=int, default=700, help='Max lines per file.')
    parser.add_argument('--warn', type=int, default=640, help='Warn band start (never fails).')
    parser.add_argument('--baseline', type=Path, default=BASELINE_PATH)
    parser.add_argument('--tighten', action='store_true', help='Lower baseline values, drop stale.')
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    violations, warnings = scan(args.root, max_lines=args.max, warn_lines=args.warn)
    baseline = _baseline.load(args.baseline)
    errors = _baseline.compare(violations, baseline, unit='lines')
    if args.tighten:
        changed = _baseline.tighten(args.baseline, violations, baseline)
        sys.stderr.write(f'tightened {changed} baseline line(s)\n')
        errors = [e for e in errors if 'new violation' in e or 'grew past' in e]

    if warnings:
        sys.stderr.write(
            f'warning: {len(warnings)} file(s) are in the {args.warn}-{args.max} line band; '
            'split them by concern when you touch them:\n'
        )
        for rel, n in sorted(warnings.items()):
            sys.stderr.write(f'  {rel}: {n}\n')
    if errors:
        sys.stderr.write(
            f'ERROR: file size ratchet failed (cap {args.max}, CLAUDE.md {CLAUDE_MD_MAX}).\n'
            'Split the file by concern; never collapse lines to fit.\n'
        )
        for e in errors:
            sys.stderr.write(f'  {e}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
