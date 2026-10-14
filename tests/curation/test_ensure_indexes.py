"""Guards against `ensure_items_*` migrations existing but never being
wired into `_ensure_indexes`.

This test enumerates every `ensure_items_*` function defined in
`curation_opensearch.ensure_fields` and `.ensure_overlay_fields` and asserts each one is actually invoked inside
`_common._migrations` — so the next unwired migration fails CI
instead of silently doing nothing.
"""

from __future__ import annotations

from src.clients.curation_opensearch import ensure_fields, ensure_overlay_fields
from src.routers.curation import _common


def test_every_ensure_migration_is_wired():
    migration_names = sorted(
        name
        for module in (ensure_fields, ensure_overlay_fields)
        for name, obj in vars(module).items()
        if name.startswith('ensure_items_') and getattr(obj, '__module__', None) == module.__name__
    )
    assert migration_names, 'expected at least one ensure_items_* export'

    wired = {migrate.__name__ for _, migrate in _common._migrations()}

    unwired = [name for name in migration_names if name not in wired]
    assert not unwired, (
        f'ensure_items_* migrations exist but are never called from _ensure_indexes: {unwired}'
    )
