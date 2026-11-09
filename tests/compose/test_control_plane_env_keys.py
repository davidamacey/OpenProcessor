"""The compose- and CLI-read control-plane settings are documented in env.template (advanced tier)."""

from __future__ import annotations

import re

import pytest

from compose._helpers import COMPOSE_PATH, REPO_ROOT


KEYS = {
    'OP_PG_MAX_CONNECTIONS': '400',
    'OP_PG_SHARED_BUFFERS': '512MB',
    'OP_JOBS_CONCURRENCY': '4',
    'OP_JOBS_MAX_MEMORY_PER_CHILD_KB': '8000000',
    'OP_VALKEY_MAXMEMORY': '1gb',
    'OP_VALKEY_MEM_LIMIT': '2g',
    'OP_PG_MEM_LIMIT': '4g',
    'OP_JOBS_MEM_LIMIT': '24g',
    'OP_PERIODIC_MEM_LIMIT': '2g',
    'OP_BEAT_MEM_LIMIT': '512m',
    'OP_STOP_GRACE_JOBS': '60',
    'OP_DB_BACKUP_KEEP': '7',
    'OP_DB_BACKUP_INTERVAL_H': '',
    'OP_DB_BACKUP_DIR': './backups/postgres',
    'OP_UID': '1000',
    'OP_GID': '1000',
}
TEMPLATE = (REPO_ROOT / 'env.template').read_text()


@pytest.mark.parametrize(('key', 'default'), sorted(KEYS.items()))
def test_each_key_has_a_template_line_with_the_documented_default(key: str, default: str) -> None:
    assert re.search(rf'^#? ?{key}={re.escape(default)}$', TEMPLATE, re.MULTILINE), key


def test_every_compose_default_matches_the_template_default() -> None:
    compose = COMPOSE_PATH.read_text()
    for key, default in KEYS.items():
        for found in re.findall(rf'\$\{{{key}:-([^}}]*)\}}', compose):
            assert found == default, f'{key}: compose {found!r} vs template {default!r}'
