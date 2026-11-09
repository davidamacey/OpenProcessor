#!/usr/bin/env python3
"""Pre-commit guard: no file lock guards cross-process state.

``api`` runs many uvicorn workers and worker containers on one shared volume. A file lock
(``fcntl.flock``/``lockf``, the ``exclusive_start_lock`` / ``exclusive_file_lock`` helpers, the
``filelock`` and ``portalocker`` packages) does not survive a second host, tells a dead holder
from a live one only by luck, and blocks a worker's thread. The cross-process claim is the fenced
claim of a ``jobs`` row (``src/db/repos/jobs.py``) or a Postgres advisory lock
(``src/config/advisory_locks.py``).

Counts, per tracked ``.py`` file under ``src scripts export docker``, the calls of those
functions and the imports of those packages (read with ``ast``, so a docstring that names one
does not count). Any use fails: there is no baseline.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

SCOPE = ('src/', 'scripts/', 'export/', 'docker/')
LOCK_HELPERS = frozenset({'exclusive_start_lock', 'exclusive_file_lock'})
FCNTL_CALLS = frozenset({'flock', 'lockf'})
LOCK_PACKAGES = frozenset({'filelock', 'portalocker'})


def tracked_files(repo_root: Path) -> list[str]:
    out = subprocess.run(  # nosec B603 B607  fixed argv, git from PATH
        ['git', 'ls-files', '-z'], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split('\0') if p]


def _uses(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', '')
            owner = func.value if isinstance(func, ast.Attribute) else None
            if name in LOCK_HELPERS or (
                name in FCNTL_CALLS and isinstance(owner, ast.Name) and owner.id == 'fcntl'
            ):
                count += 1
        elif isinstance(node, ast.Import):
            count += sum(1 for a in node.names if a.name.split('.')[0] in LOCK_PACKAGES)
        elif (
            isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] in LOCK_PACKAGES
        ):
            count += 1
    return count


def scan(repo_root: Path, files: list[str] | None = None) -> tuple[dict[str, int], list[str]]:
    """Return ``({path: uses}, unreadable files)``."""
    found: dict[str, int] = {}
    broken: list[str] = []
    for rel in files if files is not None else tracked_files(repo_root):
        if not (rel.endswith('.py') and rel.startswith(SCOPE)):
            continue
        try:
            n = _uses(ast.parse((repo_root / rel).read_text(encoding='utf-8'), filename=rel))
        except (SyntaxError, UnicodeDecodeError, OSError):
            broken.append(rel)
            continue
        if n:
            found[rel] = n
    return found, broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    found, broken = scan(args.root)
    errors = [f'{rel}: cannot be parsed, so its lock uses are unchecked' for rel in broken]
    errors += [f'{rel}: {uses} file-lock use(s)' for rel, uses in sorted(found.items())]
    if errors:
        sys.stderr.write(
            'ERROR: file lock found. Cross-process state is claimed with the fenced claim of a '
            'jobs row or a Postgres advisory lock, never a file lock.\n'
        )
        for e in errors:
            sys.stderr.write(f'  {e}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
