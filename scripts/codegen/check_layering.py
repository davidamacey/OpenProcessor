#!/usr/bin/env python3
"""Pre-commit guard: import direction between the layers of ``src/`` and ``scripts/``.

Every ``import`` and ``from ... import`` is checked, at module level and inside
functions (``PLC0415`` is ignored for lazy imports, and the cycles hide in
function bodies). Contracts:

1. ``src/{services,clients,core,config,utils,schemas}`` and ``scripts/`` (except
   ``scripts/codegen/``) must not import ``src.routers``.
2. ``src/clients`` must not import ``src.services``.
3. ``src/{services,clients,config,utils,schemas}`` must not import ``fastapi`` or
   ``starlette`` (HTTP stays in the routers).
4. ``src/{config,utils,schemas}`` must not import ``src.services`` or ``src.clients``.

Today's findings live in ``baselines/layering.txt`` as ``path -> module`` with the
number of import statements; the file can only shrink (see ``_baseline.py``).
The one permanent entry is the OpenSearch client factory, which wraps every client
in the project isolation guard at construction.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / 'scripts' / 'codegen' / 'baselines' / 'layering.txt'

# (applies to a source path prefix, forbidden module prefixes)
_NO_ROUTERS = ('src.routers',)
_NO_HTTP = ('fastapi', 'starlette')
_SRC_LAYERS_NO_ROUTERS = (
    'src/services/',
    'src/clients/',
    'src/core/',
    'src/config/',
    'src/utils/',
    'src/schemas/',
)
CONTRACTS: tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    ((*_SRC_LAYERS_NO_ROUTERS, 'scripts/'), _NO_ROUTERS, ('scripts/codegen/',)),
    (('src/clients/',), ('src.services',), ()),
    (
        ('src/services/', 'src/clients/', 'src/config/', 'src/utils/', 'src/schemas/'),
        _NO_HTTP,
        (),
    ),
    (('src/config/', 'src/utils/', 'src/schemas/'), ('src.services', 'src.clients'), ()),
)


def tracked_python(repo_root: Path) -> list[str]:
    out = subprocess.run(  # nosec B603 B607  fixed argv, git from PATH
        ['git', 'ls-files', '-z', '*.py'], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split('\0') if p.startswith(('src/', 'scripts/'))]


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + '.')


def imported_modules(rel: str, tree: ast.AST) -> list[str]:
    """Absolute module name of every import statement in ``tree`` (one entry per statement)."""
    package = rel.removesuffix('.py').split('/')[:-1]
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                modules.append('.'.join([*base, *([node.module] if node.module else [])]))
            elif node.module:
                modules.append(node.module)
    return modules


def violated(rel: str, module: str) -> bool:
    for applies, forbidden, exempt in CONTRACTS:
        if rel.startswith(applies) and not rel.startswith(exempt):
            for f in forbidden:
                if _matches(module, f):
                    return True
    return False


def scan(repo_root: Path, files: list[str] | None = None) -> tuple[dict[str, int], list[str]]:
    """Return ``({"path -> module": statements}, unparseable files)``."""
    found: dict[str, int] = {}
    broken: list[str] = []
    for rel in files if files is not None else tracked_python(repo_root):
        try:
            tree = ast.parse((repo_root / rel).read_text(encoding='utf-8'), filename=rel)
        except (SyntaxError, UnicodeDecodeError, OSError):
            broken.append(rel)
            continue
        for module in imported_modules(rel, tree):
            if violated(rel, module):
                key = f'{rel} -> {module}'
                found[key] = found.get(key, 0) + 1
    return found, broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASELINE_PATH)
    parser.add_argument('--tighten', action='store_true', help='Lower baseline values, drop stale.')
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    found, broken = scan(args.root)
    baseline = _baseline.load(args.baseline)
    errors = [f'{rel}: cannot be parsed, so its imports are unchecked' for rel in broken]
    errors += _baseline.compare(found, baseline, unit='import(s)')
    if args.tighten:
        changed = _baseline.tighten(args.baseline, found, baseline)
        sys.stderr.write(f'tightened {changed} baseline line(s)\n')
        errors = [e for e in errors if 'new violation' in e or 'grew past' in e or 'parsed' in e]
    if errors:
        sys.stderr.write(
            'ERROR: layering contract failed. Routers may import services and clients; '
            'services and clients never import routers; clients never import services; '
            'only routers import fastapi.\n'
        )
        for e in errors:
            sys.stderr.write(f'  {e}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
