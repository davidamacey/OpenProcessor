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
    ``src.services.projects.registry.projects_index()`` for ``op_projects``.

    Refuses a name that would collide with the project namespace (a
    project-index prefix, which the project guard would then treat as one
    project's index) or with the project registry itself."""
    from src.config.projects import project_index_prefix
    from src.services.projects.registry import projects_index

    name = os.environ.get('OP_GLOBAL_CONFIGS_INDEX', 'op_global_configs')
    if name.startswith(project_index_prefix()):
        msg = (
            f"OP_GLOBAL_CONFIGS_INDEX '{name}' starts with the project index prefix "
            f"'{project_index_prefix()}'; the global store must live outside every project"
        )
        raise ValueError(msg)
    if name == projects_index():
        msg = f"OP_GLOBAL_CONFIGS_INDEX '{name}' is the project registry index (OP_PROJECTS_INDEX)"
        raise ValueError(msg)
    return name


# Same config-store row shapes as a project's folded ``configs`` index
# (``src.clients.curation_opensearch.bodies_other._configs_body``), minus the folded
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
    not its I/O. While a project is bound the guard refuses every access to
    this index except a read made inside
    :func:`~src.services.projects.guard.global_configs_read` (the config
    store's own reads enter it). So :meth:`ConfigStore.refresh` called
    directly on a bound request degrades to a stale, empty snapshot, and a
    write through it (``activate_axis``) is refused by the guard before
    anything is written: the activation is neither stored nor applied to the
    in-process store. Global writes belong on unbound routes.
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
