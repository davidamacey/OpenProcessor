#!/usr/bin/env python3
"""Pre-commit regression guard: cap per-file line count.

Prevents source files from regrowing past a threshold so focused-module
splits don't silently revert. Run by the ``max-file-size`` pre-commit
hook over ``src/`` and ``scripts/``.

The threshold is set by ``--max`` (default 700). Files listed in the
hook's ``exclude`` regex are not passed in at all; this script's only
job is the line-count check.

Exit code 1 if any input file exceeds the cap; the violating files +
sizes are written to stderr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def line_count(path: Path) -> int:
    try:
        return sum(1 for _ in path.open('rb'))
    except OSError:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max', type=int, default=700, help='Max LOC per file.')
    parser.add_argument('files', nargs='*', type=Path)
    args = parser.parse_args()

    violations: list[tuple[Path, int]] = []
    for p in args.files:
        n = line_count(p)
        if n > args.max:
            violations.append((p, n))

    if violations:
        sys.stderr.write(
            f'ERROR: {len(violations)} file(s) exceed the {args.max} LOC ceiling.\n'
            'Split into focused sub-modules instead of growing monoliths.\n'
            'Violations:\n'
        )
        for p, n in violations:
            sys.stderr.write(f'  {p.as_posix()}: {n} lines (max {args.max})\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
