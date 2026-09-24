"""Guards against `ensure_items_*` migrations existing but never being
wired into `_ensure_indexes`.

This test enumerates every `ensure_items_*` name exported from
`curation_opensearch` and asserts each one is actually invoked inside
`_common._ensure_indexes` — so the next unwired migration fails CI
instead of silently doing nothing.
"""

from __future__ import annotations

import inspect
import re

from src.clients import curation_opensearch
from src.routers.curation import _common


# `ensure_items_class_name_keyword` is intentionally never wired into
# `_ensure_indexes` (matches `_common._ensure_indexes`'s own explanatory
# comment). The generic `class_name` field is declared `keyword` directly
# in `_items_body()`, so this migration's `PUT _mapping` (which tries to
# add a `.keyword` subfield to a `text` field) always fails on a fresh
# deployment; it exists only for a legacy index that still maps
# `class_name` as `text`.
_INTENTIONALLY_UNWIRED = frozenset({'ensure_items_class_name_keyword'})


def test_every_ensure_migration_is_wired():
    migration_names = sorted(
        name for name in curation_opensearch.__all__ if name.startswith('ensure_items_')
    )
    assert migration_names, 'expected at least one ensure_items_* export'

    # F-28.4: the actual bootstrap sequence moved into
    # _ensure_indexes_locked (called by _ensure_indexes while holding
    # _ensure_indexes_lock) so the lock/fast-path wiring stays readable
    # -- inspect that one for the migration calls.
    source = inspect.getsource(_common._ensure_indexes_locked)
    called_names = set(re.findall(r'\bensure_items_\w+(?=\()', source))

    unwired = [
        name
        for name in migration_names
        if name not in called_names and name not in _INTENTIONALLY_UNWIRED
    ]
    assert not unwired, (
        f'ensure_items_* migrations exist but are never called from _ensure_indexes: {unwired}'
    )
