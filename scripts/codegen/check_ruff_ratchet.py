#!/usr/bin/env python3
"""Pre-commit guard: per-function complexity and per-file broad-except counts may not grow.

Two ratchets over ``src scripts docker export models`` (third-party
``src/ultralytics_patches`` excluded), both read from ruff run with ``--isolated``
so a per-file ignore cannot hide a finding:

* ``C901`` at 15: key ``path::function`` (``#2``, ``#3`` for same-named functions in
  one file, in source order), value = McCabe score. A function over 15 must be in
  ``baselines/complexity.txt``; it may not grow.
* ``BLE001`` (bare ``except Exception``): key ``path``, value = count in the file,
  in ``baselines/broad_except.txt``; it may not grow.

Baselines can only shrink (see ``_baseline.py``); ``--tighten`` lowers them.
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
BASELINES = REPO_ROOT / 'scripts' / 'codegen' / 'baselines'
TARGETS = ('src', 'scripts', 'docker', 'export', 'models')
EXCLUDE = 'src/ultralytics_patches'
MAX_COMPLEXITY = 15
_SCORE = re.compile(r'`(?P<name>[^`]+)` is too complex \((?P<score>\d+) >')


def run_ruff(root: Path, *rule_args: str) -> list[dict[str, object]]:
    """Run ruff isolated over the targets that exist; raise on a crash or a syntax error."""
    targets = [t for t in TARGETS if (root / t).is_dir()]
    if not targets:
        return []
    proc = subprocess.run(  # nosec B603  fixed argv, our own interpreter
        [
            sys.executable,
            '-m',
            'ruff',
            'check',
            '--isolated',
            '--no-cache',
            '--output-format',
            'json',
            '--exclude',
            EXCLUDE,
            *rule_args,
            *targets,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        msg = f'ruff failed (exit {proc.returncode}): {proc.stderr.strip()}'
        raise RuntimeError(msg)
    diagnostics: list[dict[str, object]] = json.loads(proc.stdout or '[]')
    syntax = [d for d in diagnostics if d.get('code') is None]
    if syntax:
        names = ', '.join(sorted({str(d['filename']) for d in syntax}))
        msg = f'ruff could not parse: {names}'
        raise RuntimeError(msg)
    return diagnostics


def _rel(root: Path, filename: object) -> str:
    return Path(str(filename)).resolve().relative_to(root.resolve()).as_posix()


def complexity_violations(root: Path) -> dict[str, int]:
    found: dict[str, int] = {}
    rows = run_ruff(
        root,
        '--select',
        'C901',
        '--config',
        f'lint.mccabe.max-complexity={MAX_COMPLEXITY}',
    )
    rows.sort(key=lambda d: (str(d['filename']), d['location']['row']))  # type: ignore[index]
    for d in rows:
        match = _SCORE.search(str(d['message']))
        if match is None:
            continue
        base = f'{_rel(root, d["filename"])}::{match["name"]}'
        key, n = base, 1
        while key in found:
            n += 1
            key = f'{base}#{n}'
        found[key] = int(match['score'])
    return found


def broad_except_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in run_ruff(root, '--select', 'BLE001'):
        rel = _rel(root, d['filename'])
        counts[rel] = counts.get(rel, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complexity-baseline', type=Path, default=BASELINES / 'complexity.txt')
    parser.add_argument(
        '--broad-except-baseline', type=Path, default=BASELINES / 'broad_except.txt'
    )
    parser.add_argument('--tighten', action='store_true', help='Lower baseline values, drop stale.')
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    try:
        complexity = complexity_violations(args.root)
        broad = broad_except_counts(args.root)
    except RuntimeError as exc:
        sys.stderr.write(f'ERROR: {exc}\n')
        return 1

    errors: list[str] = []
    for label, current, path, unit in (
        ('complexity above 15', complexity, args.complexity_baseline, 'CC'),
        ('broad except', broad, args.broad_except_baseline, 'handler(s)'),
    ):
        baseline = _baseline.load(path)
        found = _baseline.compare(current, baseline, unit=unit)
        if args.tighten:
            changed = _baseline.tighten(path, current, baseline)
            sys.stderr.write(f'tightened {changed} line(s) in {path.name}\n')
            found = [e for e in found if 'new violation' in e or 'grew past' in e]
        errors += [f'[{label}] {e}' for e in found]

    if errors:
        sys.stderr.write(
            'ERROR: complexity / broad-except ratchet failed. Decompose the function or '
            'catch the specific exception; the baselines only shrink.\n'
        )
        for e in errors:
            sys.stderr.write(f'  {e}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
