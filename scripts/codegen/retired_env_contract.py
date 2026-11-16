#!/usr/bin/env python3
"""Render ``contracts/json/retired_env.json``: the environment variables a process refuses to start with.

``src/config/retired_env.py`` stays the one definition (``RETIRED_ENV``). Every role of the Rust
server reads this file (``native/op_config``, module ``retired``) and refuses to start when any
listed name is set, with the same words ``reject_retired_env`` uses: ``<name> was <was>``, where
``<was>`` is ``renamed to <new name>`` or ``removed: <what to use instead>``.

The file lists the names sorted, one object each:

    {"name": "OP_TRITON_URL", "was": "renamed to TRITON_URL"}

``--check`` exits 1 when the committed file differs from what the table renders (a pre-commit hook).

Usage:

    python3 scripts/codegen/retired_env_contract.py          # write
    python3 scripts/codegen/retired_env_contract.py --check  # drift check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _contract_runtime


REPO_ROOT = _contract_runtime.REPO_ROOT
TARGET = REPO_ROOT / 'contracts' / 'json' / 'retired_env.json'
SCRIPT_REL = 'scripts/codegen/retired_env_contract.py'
SOURCE_REL = 'src/config/retired_env.py'
_REMOVED = 'removed:'


def was(replacement: str) -> str:
    """The words after ``<name> was`` in the refusal: the table's own text for a removed setting,
    ``renamed to <new name>`` for a rename (as ``reject_retired_env`` writes them)."""
    return replacement if replacement.startswith(_REMOVED) else f'renamed to {replacement}'


def render() -> str:
    """The file's text for the table as it stands."""
    from src.config.retired_env import (
        RETIRED_ENV,  # imports the app; runs in the sandbox child only
    )

    document = {
        'generated_by': SCRIPT_REL,
        'source': SOURCE_REL,
        'retired': [{'name': name, 'was': was(RETIRED_ENV[name])} for name in sorted(RETIRED_ENV)],
    }
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


def run(target: Path, *, check_only: bool) -> int:
    text = render()
    if not check_only:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
        sys.stdout.write(f'wrote {target}\n')
        return 0
    if not target.exists() or target.read_text(encoding='utf-8') != text:
        sys.stderr.write(
            f'{target} is out of sync with {SOURCE_REL}; '
            f're-run: python3 {SCRIPT_REL} (or make contracts)\n'
        )
        return 1
    sys.stdout.write(f'{target} is in sync\n')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--check', action='store_true', help='Do not write; exit 1 on drift.')
    parser.add_argument('--target', type=Path, default=TARGET, help='File to write or check.')
    args = parser.parse_args(argv)
    if not _contract_runtime.in_sandbox():
        child = ['--target', str(args.target.resolve())] + (['--check'] if args.check else [])
        result: int = _contract_runtime.reexec(Path(__file__).resolve(), child)
        return result
    return run(args.target, check_only=args.check)


if __name__ == '__main__':
    sys.exit(main())
