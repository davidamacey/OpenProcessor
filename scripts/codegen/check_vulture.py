#!/usr/bin/env python3
"""Pre-commit guard: no new unused code (vulture, min confidence 60) in production sources.

Runs vulture over ``src scripts export docker`` (tests are not scanned, so code
used only by tests is reported: production code with no production caller is dead).
Route handlers and validators are registered by decorator and are ignored; fields
declared in a class body (pydantic and dataclass fields, enum members) are data and
are dropped from the report. A reviewed false positive goes into
``scripts/codegen/vulture_whitelist.py`` with a reason on its line.

Findings that existed when the guard landed live in ``baselines/vulture.txt`` as
``path::kind name``; the file can only shrink (see ``_baseline.py``).
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / 'scripts' / 'codegen' / 'baselines' / 'vulture.txt'
TARGETS = ('src', 'scripts', 'export', 'docker')
EXCLUDE = '*/ultralytics_patches/*'
MIN_CONFIDENCE = '60'
IGNORE_DECORATORS = (
    '@*router.*,@app.*,@application.*,@*.on_event,@field_validator,@model_validator,@validator,'
    '@computed_field,@asynccontextmanager,@pytest.fixture,@abstractmethod,@*.setter'
)
_FINDING = re.compile(r"^(?P<path>.+?):(?P<line>\d+): unused (?P<kind>\w+) '(?P<name>.+)' \(")


def _class_body_lines(source_path: Path) -> set[int]:
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    return {
        stmt.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign | ast.Assign)
    }


def findings(root: Path) -> dict[str, int]:
    targets = [t for t in TARGETS if (root / t).is_dir()]
    if not targets:
        return {}
    proc = subprocess.run(  # nosec B603  fixed argv, our own interpreter
        [
            sys.executable,
            '-m',
            'vulture',
            *targets,
            '--min-confidence',
            MIN_CONFIDENCE,
            '--exclude',
            EXCLUDE,
            '--ignore-decorators',
            IGNORE_DECORATORS,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 3):
        msg = (
            f'vulture failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}'
        )
        raise RuntimeError(msg)
    body_lines: dict[str, set[int]] = {}
    found: dict[str, int] = {}
    for raw in proc.stdout.splitlines():
        match = _FINDING.match(raw)
        if match is None:
            continue
        path, kind, name = match['path'], match['kind'], match['name']
        if kind in ('variable', 'attribute'):
            if path not in body_lines:
                body_lines[path] = _class_body_lines(root / path)
            if int(match['line']) in body_lines[path]:
                continue
        key = f'{path}::{kind} {name}'
        found[key] = found.get(key, 0) + 1
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASELINE_PATH)
    parser.add_argument('--tighten', action='store_true', help='Lower baseline values, drop stale.')
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    try:
        found = findings(args.root)
    except (RuntimeError, SyntaxError) as exc:
        sys.stderr.write(f'ERROR: {exc}\n')
        return 1
    baseline = _baseline.load(args.baseline)
    errors = _baseline.compare(found, baseline, unit='finding(s)')
    if args.tighten:
        changed = _baseline.tighten(args.baseline, found, baseline)
        sys.stderr.write(f'tightened {changed} baseline line(s)\n')
        errors = [e for e in errors if 'new violation' in e or 'grew past' in e]
    if errors:
        sys.stderr.write(
            'ERROR: unused code. Delete it, or if a framework calls it by name add it to '
            'scripts/codegen/vulture_whitelist.py with the reason.\n'
        )
        for e in errors:
            sys.stderr.write(f'  {e}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
