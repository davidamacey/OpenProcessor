#!/usr/bin/env python3
"""Pre-commit guard: duplicated code (copy/paste clones) may not grow.

Runs jscpd (https://github.com/kucherenko/jscpd, the maintained multi-language clone
detector; the Python and the Svelte/TypeScript code are scanned in one pass) over
``src scripts frontend/src`` and keys each clone group by its sorted file pair.
The value is the duplicated line count of that pair, in ``baselines/duplication.txt``.
A clone between two files that is not in the baseline, or a baselined pair whose
duplicated lines grew, fails. Baselines can only shrink (see ``_baseline.py``).

Minimum clone: 12 lines and 100 tokens. Test files are not scanned: fixtures there
are repetitive by design and are covered by the shared-fixture rule in review.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / 'scripts' / 'codegen' / 'baselines' / 'duplication.txt'
JSCPD = Path('frontend') / 'node_modules' / '.bin' / 'jscpd'
TARGETS = ('src', 'scripts', 'frontend/src')
FORMATS = 'python,typescript,javascript,svelte'
MIN_LINES = '12'
MIN_TOKENS = '100'
IGNORE = ','.join(
    (
        '**/node_modules/**',
        '**/__pycache__/**',
        '**/*.test.ts',
        '**/*.spec.ts',
        '**/*.d.ts',
    )
)


def _rel(root: Path, name: str) -> str:
    return Path(name).resolve().relative_to(root.resolve()).as_posix()


def clone_pairs(root: Path) -> dict[str, int]:
    """Duplicated lines per ``fileA <-> fileB`` pair (same-file clones: ``fileA <-> fileA``)."""
    binary = root / JSCPD
    if not binary.exists():
        msg = f'{binary} missing: run `npm ci` in frontend/'
        raise RuntimeError(msg)
    targets = [t for t in TARGETS if (root / t).is_dir()]
    with tempfile.TemporaryDirectory() as out:
        proc = subprocess.run(  # nosec B603  fixed argv
            [
                str(binary),
                *targets,
                '--format',
                FORMATS,
                '--min-lines',
                MIN_LINES,
                '--min-tokens',
                MIN_TOKENS,
                '--ignore',
                IGNORE,
                '--reporters',
                'json',
                '--output',
                out,
                '--absolute',
                '--silent',
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        report = Path(out) / 'jscpd-report.json'
        if proc.returncode not in (0, 1) or not report.exists():
            msg = f'jscpd failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}'
            raise RuntimeError(msg)
        data = json.loads(report.read_text(encoding='utf-8'))
    pairs: dict[str, int] = {}
    for dup in data['duplicates']:
        names = sorted(_rel(root, dup[side]['name']) for side in ('firstFile', 'secondFile'))
        key = f'{names[0]} <-> {names[1]}'
        pairs[key] = pairs.get(key, 0) + int(dup['lines'])
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--tighten', action='store_true', help='Lower baseline values, drop stale.')
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    try:
        current = clone_pairs(args.root)
    except RuntimeError as exc:
        sys.stderr.write(f'ERROR: {exc}\n')
        return 1

    baseline = _baseline.load(args.baseline)
    errors = _baseline.compare(current, baseline, unit='duplicated lines')
    if args.tighten:
        changed = _baseline.tighten(args.baseline, current, baseline)
        sys.stderr.write(f'tightened {changed} line(s) in {args.baseline.name}\n')
        errors = [e for e in errors if 'new violation' in e or 'grew past' in e]
    if errors:
        sys.stderr.write(
            'ERROR: duplicated code. Extract one shared function, component or module and '
            'use it from both places; the baseline only shrinks.\n'
        )
        for e in errors:
            sys.stderr.write(f'  {e}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
