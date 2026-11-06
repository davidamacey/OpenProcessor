#!/usr/bin/env python3
"""Pre-commit guard: import direction between the layers of ``src/`` and ``scripts/``.

Every ``import`` and ``from ... import`` is checked, at module level and inside
functions (``PLC0415`` is ignored for lazy imports, and the cycles hide in
function bodies). Contracts:

1. ``src/{services,clients,core,config,domain,utils,schemas}`` and ``scripts/`` (except
   ``scripts/codegen/``) must not import ``src.routers``.
2. ``src/clients`` must not import ``src.services``.
3. ``src/{services,clients,config,domain,utils,schemas}`` must not import ``fastapi`` or
   ``starlette`` (HTTP stays in the routers).
4. ``src/{config,domain,utils,schemas}`` must not import ``src.services`` or ``src.clients``.
5. ``src/db`` must not import ``src.services``, ``src.clients``, ``src.routers``, ``fastapi`` or
   ``starlette``; ``src/{config,domain,utils,schemas}`` must not import ``src.db``; and
   ``src/routers`` must not import ``src.db.repos`` (handlers go through services).
6. ``src/tasks`` must not import ``src.routers``, ``fastapi`` or ``starlette``.
7. ``src.tasks.runtime`` is a leaf layer. Anything outside ``src/tasks/`` (services, scripts,
   the trainer, export, the bake-off) may import it and no other task module, and
   ``src/tasks/runtime/`` itself imports only ``src.db``, ``src.config``, ``src.domain``,
   ``src.utils`` and itself, so the trainer, export and evaluator images can carry it without
   FastAPI, OpenSearch or torch.

There is no baseline: a violation fails. The one permanent exception is
:data:`PERMANENT_EXCEPTIONS`, with its reason; an exception whose import is gone
fails as stale.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# ``path -> module`` : reason. Keep this list to what cannot move.
PERMANENT_EXCEPTIONS: dict[str, str] = {
    'src/clients/opensearch/client.py -> src.services.projects.guard': (
        'the guarded OpenSearch client factory wraps every client in project isolation at '
        'construction'
    ),
    'src/clients/opensearch/client.py -> src.services.projects.registry': (
        'the guarded OpenSearch client factory wraps every client in project isolation at '
        'construction'
    ),
}

# (applies to a source path prefix, forbidden module prefixes)
_NO_ROUTERS = ('src.routers',)
_NO_HTTP = ('fastapi', 'starlette')
_SRC_LAYERS_NO_ROUTERS = (
    'src/services/',
    'src/clients/',
    'src/core/',
    'src/config/',
    'src/domain/',
    'src/utils/',
    'src/schemas/',
)
CONTRACTS: tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    ((*_SRC_LAYERS_NO_ROUTERS, 'scripts/'), _NO_ROUTERS, ('scripts/codegen/',)),
    (('src/clients/',), ('src.services',), ()),
    (
        (
            'src/services/',
            'src/clients/',
            'src/config/',
            'src/domain/',
            'src/utils/',
            'src/schemas/',
        ),
        _NO_HTTP,
        (),
    ),
    (
        ('src/config/', 'src/domain/', 'src/utils/', 'src/schemas/'),
        ('src.services', 'src.clients'),
        (),
    ),
    (('src/db/',), ('src.services', 'src.clients', 'src.routers', *_NO_HTTP), ()),
    (('src/config/', 'src/domain/', 'src/utils/', 'src/schemas/'), ('src.db',), ()),
    (('src/routers/',), ('src.db.repos',), ()),
    (('src/tasks/',), ('src.routers', *_NO_HTTP), ()),
)

# The only ``src`` packages ``src/tasks/runtime/`` may import.
_RUNTIME_MAY_IMPORT = ('src.db', 'src.config', 'src.domain', 'src.utils', 'src.tasks.runtime')


def tracked_python(repo_root: Path) -> list[str]:
    out = subprocess.run(  # nosec B603 B607  fixed argv, git from PATH
        ['git', 'ls-files', '-z', '*.py'], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split('\0') if p.startswith(('src/', 'scripts/', 'docker/', 'export/'))]


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


def _breaks_task_layers(rel: str, module: str) -> bool:
    if rel.startswith('src/tasks/runtime/') and (module == 'src' or module.startswith('src.')):
        return not any(_matches(module, ok) for ok in _RUNTIME_MAY_IMPORT)
    if not rel.startswith('src/tasks/') and _matches(module, 'src.tasks'):
        return not _matches(module, 'src.tasks.runtime')
    return False


def violated(rel: str, module: str) -> bool:
    if _breaks_task_layers(rel, module):
        return True
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
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    found, broken = scan(args.root)
    errors = [f'{rel}: cannot be parsed, so its imports are unchecked' for rel in broken]
    errors += [
        f'{key}: {count} import(s) cross a layer'
        for key, count in found.items()
        if key not in PERMANENT_EXCEPTIONS
    ]
    errors += [
        f'{key}: stale exception, the import is gone (delete it from PERMANENT_EXCEPTIONS)'
        for key in PERMANENT_EXCEPTIONS
        if key not in found
    ]
    if errors:
        sys.stderr.write(
            'ERROR: layering contract failed. Routers may import services and clients; '
            'services and clients never import routers; clients never import services; '
            'only routers import fastapi; src/db imports no service, client or router; '
            'src/tasks/runtime is a leaf that only db, config, domain and utils may feed.\n'
        )
        for e in errors:
            sys.stderr.write(f'  {e}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
