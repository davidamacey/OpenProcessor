"""The global (non-project-scoped) config store: ``op_global_configs`` (M3).

Everything in :mod:`src.services.config_store.store` is per-project (keyed
by the bound project's slug, unusable unbound). This is its sibling: ONE
store for the whole deployment, for a config axis that is not scoped to any
project -- W9's VLM endpoint registry (endpoint docs, revision copies, probe
docs and the desired local model), the same way ``op_projects``
(src.services.projects.registry) is the one other index that lives outside
every project's own index set.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Literal

from src.services.config_store.store import ConfigStore


def global_configs_index() -> str:
    """The one config-store index that belongs to no project -- mirrors
    ``src.services.projects.registry.projects_index()`` for ``op_projects``."""
    return os.environ.get('OP_GLOBAL_CONFIGS_INDEX', 'op_global_configs')


# Same config-store row shapes as a project's folded ``configs`` index
# (``src.clients.curation_opensearch._configs_body``), minus the folded
# SETTINGS/UMAP_VIZ_STATE properties -- the global store never folds any
# other role onto it, so it carries only the config-store fields
# ``src.services.config_store.index``'s primitives read/write (``get``,
# ``save_config``, ``activate``, ``upsert_runtime_doc``, ...). Kept as its
# own literal body (not imported from ``curation_opensearch``) so this
# module stays dependency-light, per its own module docstring.
GLOBAL_CONFIGS_INDEX_BODY: dict[str, Any] = {
    'settings': {'index': {'number_of_shards': 1, 'number_of_replicas': 0}},
    'mappings': {
        'dynamic': False,
        'properties': {
            'doc_type': {'type': 'keyword'},
            'kind': {'type': 'keyword'},
            'name': {'type': 'keyword'},
            'revision': {'type': 'integer'},
            'body': {'type': 'object', 'enabled': False},
            'description': {'type': 'keyword', 'ignore_above': 512, 'index': False},
            'created_at': {'type': 'date'},
            'updated_at': {'type': 'date'},
            'updated_by': {'type': 'keyword'},
            'cloned_from': {'type': 'keyword'},
            'axis': {'type': 'keyword'},
            'previous': {'type': 'object', 'enabled': False},
            'config_revision': {'type': 'long'},
            'process': {'type': 'keyword'},
            'applied_at': {'type': 'date'},
        },
    },
}


async def ensure_global_configs_index(client: Any) -> None:
    """Create ``op_global_configs`` with its explicit mapping if it does
    not exist yet -- mirrors
    ``src.services.projects.bootstrap.ensure_projects_index`` for
    ``op_projects``. Idempotent; call at startup before the first
    global-store read/write. Needs no project bound (this index belongs
    to none)."""
    index = global_configs_index()
    if await client.indices.exists(index=index):
        return
    await client.indices.create(index=index, body=GLOBAL_CONFIGS_INDEX_BODY)


# Cached separately from `_STORES` (never by the same key/dict) so a
# global store instance can never collide with, or be mistaken for, any
# project's own store.
_GLOBAL_STORE: dict[str, ConfigStore] = {}
_GLOBAL_STORE_LOCK = threading.Lock()
_GLOBAL_STORE_KEY = 'op_global_configs'


def get_global_config_store(*, mode: Literal['live', 'pinned'] = 'live') -> ConfigStore:
    """The process's singleton :class:`ConfigStore` for
    ``op_global_configs``.

    Unlike :func:`get_config_store` (keyed by, and unusable without, the
    *bound* project), this never consults
    :func:`~src.config.project_context.current_project` at all -- it
    requires no project binding, and calling it while a project happens
    to be bound has no effect on which store it returns.

    That binding-independence is about which *store object* comes back,
    not its I/O: the object returned here still needs an unbound client
    to actually read/write (m2, W2-finish review) -- calling
    :meth:`ConfigStore.refresh` on it while a project is bound gets
    refused by the project guard (this index is unowned, so a bound
    request has no business touching it) and silently degrades to a
    stale, empty snapshot, the same as any other refresh failure. W9
    (the first real consumer with project-bound call sites) must decide
    the read-while-bound rule -- an unbound-read helper, or a guard
    exception allowing read-only access the way ``op_projects`` gets it.
    """
    with _GLOBAL_STORE_LOCK:
        store = _GLOBAL_STORE.get(_GLOBAL_STORE_KEY)
        if store is None:
            store = ConfigStore(
                index=global_configs_index(), mode=mode, label='__global__', is_global=True
            )
            _GLOBAL_STORE[_GLOBAL_STORE_KEY] = store
        return store


def reset_global_config_store() -> None:
    """Test-only: drop the cached global store so a test's fake
    OpenSearch starts from a clean snapshot."""
    from src.services.config_store.vlm_snapshot import reset_vlm_revision_cache

    with _GLOBAL_STORE_LOCK:
        _GLOBAL_STORE.clear()
    reset_vlm_revision_cache()


__all__ = [
    'GLOBAL_CONFIGS_INDEX_BODY',
    'ensure_global_configs_index',
    'get_global_config_store',
    'global_configs_index',
    'reset_global_config_store',
]
