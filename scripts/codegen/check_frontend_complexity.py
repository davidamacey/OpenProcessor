#!/usr/bin/env python3
"""Pre-commit guard: per-function complexity in frontend/src may not grow.

The frontend twin of the complexity half of ``check_ruff_ratchet.py``. ESLint's core
``complexity`` rule (https://eslint.org/docs/latest/rules/complexity, max 15, set in
``frontend/eslint.complexity.config.js`` so the normal lint stays unchanged) reports
every function over 15. Key ``path::function`` (``#2``... for repeats, in source
order; ``<anonymous>`` for unnamed functions), value = the score, in
``baselines/frontend_complexity.txt``. Baselines can only shrink (see ``_baseline.py``).

With file arguments (the commit hook) only those files are scanned and only their
baseline entries are checked for staleness; with none, all of ``frontend/src``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / 'scripts' / 'codegen' / 'baselines' / 'frontend_complexity.txt'
_MSG = re.compile(r'^(?P<what>.+?) has a complexity of (?P<score>\d+)\.')
_NAME = re.compile(r"'([^']+)'")


def scan(root: Path, files: list[str]) -> dict[str, int]:
    frontend = root / 'frontend'
    eslint = frontend / 'node_modules' / '.bin' / 'eslint'
    if not eslint.exists():
        msg = f'{eslint} missing: run `npm ci` in frontend/'
        raise RuntimeError(msg)
    targets = files or ['src']
    proc = subprocess.run(  # nosec B603  fixed argv
        [
            str(eslint),
            '-c',
            'eslint.complexity.config.js',
            '--no-warn-ignored',
            '-f',
            'json',
            *targets,
        ],
        cwd=frontend,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        msg = f'eslint failed (exit {proc.returncode}): {proc.stderr.strip()}'
        raise RuntimeError(msg)
    found: dict[str, int] = {}
    for entry in sorted(json.loads(proc.stdout or '[]'), key=lambda e: e['filePath']):
        rel = Path(entry['filePath']).resolve().relative_to(root.resolve()).as_posix()
        for m in sorted(entry['messages'], key=lambda m: (m['line'], m['column'])):
            if m.get('ruleId') != 'complexity':
                if m.get('fatal'):
                    msg = f'eslint could not parse {rel}: {m["message"]}'
                    raise RuntimeError(msg)
                continue
            match = _MSG.match(m['message'])
            if match is None:
                continue
            named = _NAME.search(match['what'])
            base = f'{rel}::{named[1] if named else "<anonymous>"}'
            key, n = base, 1
            while key in found:
                n += 1
                key = f'{base}#{n}'
            found[key] = int(match['score'])
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', help='frontend/-prefixed files (default: all of src)')
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--tighten', action='store_true', help='Lower baseline values, drop stale.')
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    files = [f.removeprefix('frontend/') for f in args.files]
    try:
        current = scan(args.root, files)
    except RuntimeError as exc:
        sys.stderr.write(f'ERROR: {exc}\n')
        return 1

    baseline = _baseline.load(args.baseline)
    if files:
        scanned = {f'frontend/{f}' for f in files}
        baseline = {k: v for k, v in baseline.items() if k.split('::', 1)[0] in scanned}
    errors = _baseline.compare(current, baseline, unit='CC')
    if args.tighten and not files:
        changed = _baseline.tighten(args.baseline, current, baseline)
        sys.stderr.write(f'tightened {changed} line(s) in {args.baseline.name}\n')
        errors = [e for e in errors if 'new violation' in e or 'grew past' in e]
    if errors:
        sys.stderr.write(
            'ERROR: frontend complexity above 15. Decompose the function; the baseline only shrinks.\n'
        )
        for e in errors:
            sys.stderr.write(f'  {e}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
