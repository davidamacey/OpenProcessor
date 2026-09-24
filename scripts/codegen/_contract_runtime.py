"""Run an app-importing codegen script in a clean, reproducible process.

Contract generators that need the FastAPI app (OpenAPI, the ``ItemDoc``
JSON schema) must not depend on who runs them: a developer shell with
``OP_API_PREFIX`` exported, or a ``.env`` in the working directory, would
otherwise bake one deployment's settings into the committed contract.
``reexec`` re-runs the calling script under the project interpreter with
a whitelisted environment and a throwaway working directory, so the
output is the same for everyone. It also lets a pre-commit hook start
from a bare system ``python3`` that lacks the app's dependencies.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# Set in the child so the script knows it is already sandboxed.
SANDBOX_ENV = 'CONTRACTS_CODEGEN_SANDBOX'
# Escape hatch: point at any interpreter that has the app's dependencies.
PYTHON_OVERRIDE_ENV = 'CONTRACTS_PYTHON'

_PASSTHROUGH_ENV = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR')


def in_sandbox() -> bool:
    return os.environ.get(SANDBOX_ENV) == '1'


def _has_app_deps() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ('fastapi', 'pydantic'))


def _main_worktree_root() -> Path | None:
    # A linked git worktree shares the main checkout's .venv.
    try:
        out = subprocess.run(  # nosec B603 B607 — fixed argv, no shell
            [
                'git',
                '-C',
                str(REPO_ROOT),
                'rev-parse',
                '--path-format=absolute',
                '--git-common-dir',
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(out).parent if out else None


def resolve_python() -> str:
    """Interpreter that can import the app, or raise with what was tried."""
    override = os.environ.get(PYTHON_OVERRIDE_ENV)
    if override:
        return override
    if _has_app_deps():
        return sys.executable
    candidates: list[Path] = []
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        candidates.append(Path(venv) / 'bin' / 'python')
    candidates.append(REPO_ROOT / '.venv' / 'bin' / 'python')
    main_root = _main_worktree_root()
    if main_root is not None:
        candidates.append(main_root / '.venv' / 'bin' / 'python')
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    tried = ', '.join(str(c) for c in candidates)
    raise SystemExit(
        f'no interpreter with the app dependencies found (tried {tried}); '
        f'set {PYTHON_OVERRIDE_ENV}=/path/to/python'
    )


def reexec(script: Path, argv: list[str]) -> int:
    """Run ``script argv`` under the project interpreter in a clean env."""
    env = {k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}
    env.update({SANDBOX_ENV: '1', 'PYTHONPATH': str(REPO_ROOT), 'PYTHONHASHSEED': '0'})
    with tempfile.TemporaryDirectory(prefix='contracts-codegen-') as cwd:
        proc = subprocess.run(  # nosec B603 — argv built from resolved paths, no shell
            [resolve_python(), str(script), *argv], cwd=cwd, env=env, check=False
        )
    return proc.returncode
