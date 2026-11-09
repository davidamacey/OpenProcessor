#!/usr/bin/env python3
"""Pre-commit guard: the list of test modules that mypy still skips may only shrink.

``tests/`` is type-checked by the one mypy hook (``[tool.mypy] files`` in ``pyproject.toml``).
Test modules that still have errors are listed in a ``[[tool.mypy.overrides]]`` block with
``ignore_errors = true`` (the per-module mechanism in the mypy docs, "Using mypy with an existing
codebase": https://mypy.readthedocs.io/en/stable/existing_code.html). A module not in the list is
fully checked, so a new test module with errors fails the mypy hook.

This guard keeps that list honest against ``baselines/mypy_tests.txt`` (``module<TAB>1<TAB>reason``):
the pyproject list and the baseline must hold the same modules, and every listed module must still
exist as a file. Fixing a module means deleting it from both; adding one needs a reason line in the
baseline that review sees. ``--tighten`` drops entries whose file is gone from both places.

A listed module that is already clean would stay listed forever, because ``ignore_errors`` hides
the proof. So one mypy process re-checks only the listed modules with ``ignore_errors`` switched
off in a temporary copy of the config (per-module options are config-file only, see
https://mypy.readthedocs.io/en/stable/config_file.html#confval-ignore_errors) and a module with
no error is reported as ``clean but still listed``. An empty list skips mypy altogether. When the
list is empty the override block is absent from ``pyproject.toml``; to skip a module again add a
``[[tool.mypy.overrides]]`` block with ``module = [...]`` and ``ignore_errors = true``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / 'scripts' / 'codegen' / 'baselines' / 'mypy_tests.txt'
PYPROJECT = REPO_ROOT / 'pyproject.toml'


def listed_modules(pyproject: Path) -> list[str]:
    """Modules of the override block that sets ``ignore_errors = true`` for tests."""
    data = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    found: list[str] = []
    for block in data['tool']['mypy']['overrides']:
        if block.get('ignore_errors') is True:
            modules = block['module']
            found.extend([modules] if isinstance(modules, str) else modules)
    return found


def module_file(repo_root: Path, module: str) -> Path:
    return repo_root / (module.replace('.', '/') + '.py')


MypyRunner = Callable[[Path, Path, list[Path]], str]
_ERROR_LINE = re.compile(r'^(?P<path>[^:\s][^:]*\.py):\d+(?::\d+)?: error:', re.MULTILINE)


def run_mypy(repo_root: Path, config: Path, files: list[Path]) -> str:
    """One mypy process over ``files`` with ``config``; its own cache dir because the changed
    config would otherwise invalidate the cache the mypy hook uses."""
    cache = Path(tempfile.gettempdir()) / 'op_mypy_tests_ignore_cache'
    done = subprocess.run(
        [sys.executable, '-m', 'mypy', '--config-file', str(config), '--cache-dir', str(cache)]
        + [str(f) for f in files],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout


def clean_but_listed(repo_root: Path, pyproject: Path, runner: MypyRunner = run_mypy) -> list[str]:
    """Listed modules that have no mypy error once ``ignore_errors`` is off."""
    modules = [m for m in listed_modules(pyproject) if module_file(repo_root, m).exists()]
    if not modules:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / 'pyproject.toml'
        config.write_text(
            pyproject.read_text(encoding='utf-8').replace(
                'ignore_errors = true', 'ignore_errors = false'
            ),
            encoding='utf-8',
        )
        output = runner(repo_root, config, [module_file(repo_root, m) for m in modules])
    failing = {
        (repo_root / m['path'])
        .resolve()
        .relative_to(repo_root.resolve())
        .with_suffix('')
        .as_posix()
        for m in _ERROR_LINE.finditer(output)
    }
    return [
        f'{m}: listed in the ignore_errors block but has no mypy error (delete it from '
        'pyproject.toml and the baseline)'
        for m in modules
        if m.replace('.', '/') not in failing
    ]


def check(repo_root: Path, pyproject: Path, baseline_path: Path) -> list[str]:
    baseline = _baseline.load(baseline_path)
    listed = listed_modules(pyproject)
    errors: list[str] = []
    if len(listed) != len(set(listed)):
        errors.append('pyproject.toml lists a test module twice in the ignore_errors block')
    errors.extend(
        f'{m}: ignored in pyproject.toml but not in baselines/mypy_tests.txt (new debt)'
        for m in sorted(set(listed) - set(baseline))
    )
    errors.extend(
        f'{m}: in baselines/mypy_tests.txt but not ignored in pyproject.toml (delete the line)'
        for m in sorted(set(baseline) - set(listed))
    )
    errors.extend(
        f'{m}: listed but {module_file(repo_root, m).name} no longer exists (delete the entry)'
        for m in sorted(set(listed) | set(baseline))
        if not module_file(repo_root, m).exists()
    )
    errors.extend(
        f'{m}: only tests.* modules may be ignored'
        for m in sorted(listed)
        if not m.startswith('tests.')
    )
    return errors


def tighten(repo_root: Path, pyproject: Path, baseline_path: Path) -> int:
    gone = {m for m in listed_modules(pyproject) if not module_file(repo_root, m).exists()}
    gone |= {m for m in _baseline.load(baseline_path) if not module_file(repo_root, m).exists()}
    text = pyproject.read_text(encoding='utf-8')
    for m in gone:
        text = text.replace(f'    "{m}",\n', '')
    pyproject.write_text(text, encoding='utf-8')
    kept = [
        line
        for line in baseline_path.read_text(encoding='utf-8').splitlines()
        if line.split('\t')[0] not in gone
    ]
    baseline_path.write_text('\n'.join(kept) + '\n', encoding='utf-8')
    return len(gone)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tighten', action='store_true', help='Drop entries whose file is gone.')
    args = parser.parse_args(argv)
    if args.tighten:
        sys.stderr.write(f'dropped {tighten(REPO_ROOT, PYPROJECT, BASELINE)} entries\n')
        return 0
    errors = check(REPO_ROOT, PYPROJECT, BASELINE)
    if not errors:
        errors = clean_but_listed(REPO_ROOT, PYPROJECT)
    for line in errors:
        sys.stderr.write(line + '\n')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
