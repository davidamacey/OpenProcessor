"""Whole-stack bring-ups must pass ``--remove-orphans``.

The compose service was renamed ``yolo-api`` -> ``api`` while keeping the same
``container_name``. Without ``--remove-orphans`` the old container (its service
label no longer exists) stays and the new ``api`` container fails with a name
conflict, so every path that starts the stack on an existing install has to
remove orphans.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# file -> regex matching the whole-stack `up` lines that must carry the flag
_WHOLE_STACK_UP = {
    'openprocessor': r'^\s*dc up -d( --remove-orphans)?$',
    'scripts/setup.sh': r'^\s*dc up -d( --remove-orphans)?$',
    'Makefile': r'^\t\$\((DEV_)?COMPOSE\) (--profile monitoring )?up -d( --build)?( --remove-orphans)?$',
    'setup-openprocessor.sh': r'dc up -d( --remove-orphans)?( opensearch)?( \|\||$)',
}


def test_whole_stack_up_commands_remove_orphans() -> None:
    for path, pattern in _WHOLE_STACK_UP.items():
        rx = re.compile(pattern)
        found = [
            ln.strip()
            for ln in (ROOT / path).read_text().splitlines()
            if rx.search(ln) and not ln.lstrip().startswith('#')
        ]
        assert found, f'{path}: pattern matched no up command; update this test'
        missing = [ln for ln in found if '--remove-orphans' not in ln]
        assert not missing, f'{path}: up without --remove-orphans: {missing}'
