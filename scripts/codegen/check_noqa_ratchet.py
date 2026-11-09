#!/usr/bin/env python3
"""Pre-commit guard: the number of noqa comments per Python file may not grow.

A noqa marker is an exception, not a shortcut. Every tracked Python file with one is
listed in ``baselines/noqa.txt`` (key = path, value = count of lines carrying
the noqa marker). A new suppression needs a baseline bump that review sees; the
baseline can only shrink otherwise (see ``_baseline.py``); ``--tighten`` lowers it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = REPO_ROOT / 'scripts' / 'codegen' / 'baselines' / 'noqa.txt'
_NOQA = re.compile(r'#\s*noqa\b', re.IGNORECASE)
_PLACEHOLDER = 'suppressions recorded when the ratchet landed; fix or justify, then lower'


def noqa_counts(root: Path) -> dict[str, int]:
    """Lines carrying the noqa marker per tracked Python file."""
    listing = subprocess.run(  # nosec B603 B607  fixed argv
        ['git', 'ls-files', '-z', '--', '*.py'],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    counts: dict[str, int] = {}
    for rel in filter(None, listing.split('\0')):
        path = root / rel
        if not path.is_file():
            continue
        n = sum(1 for line in path.read_text(encoding='utf-8').splitlines() if _NOQA.search(line))
        if n:
            counts[rel] = n
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--tighten', action='store_true', help='Lower baseline values, drop stale.')
    parser.add_argument('--init', action='store_true', help='Write a fresh baseline (one-time).')
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    current = noqa_counts(args.root)
    if args.init:
        args.baseline.write_text(
            ''.join(f'{k}\t{v}\t{_PLACEHOLDER}\n' for k, v in sorted(current.items())),
            encoding='utf-8',
        )
        return 0
    baseline = _baseline.load(args.baseline)
    errors = _baseline.compare(current, baseline, unit='noqa line(s)')
    if args.tighten:
        changed = _baseline.tighten(args.baseline, current, baseline)
        sys.stderr.write(f'tightened {changed} line(s) in {args.baseline.name}\n')
        errors = [e for e in errors if 'new violation' in e or 'grew past' in e]
    if errors:
        sys.stderr.write(
            'ERROR: noqa ratchet failed. Fix the code instead of suppressing the rule; '
            'the baseline only shrinks.\n'
        )
        for e in errors:
            sys.stderr.write(f'  {e}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
