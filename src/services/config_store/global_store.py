"""The global (non-project-scoped) config store.

Everything in :mod:`src.services.config_store.store` is per-project (keyed by the bound
project's slug, unusable unbound). This is its sibling: ONE store for the whole deployment, for a
config axis that is not scoped to any project: the VLM endpoint registry (endpoint documents,
each revision's last probe and the desired local model).
"""

from __future__ import annotations

import threading
from typing import Literal

from src.services.config_store.store import ConfigStore


# Cached separately from the project stores (never by the same key/dict) so a global store
# instance can never collide with, or be mistaken for, any project's own store.
_GLOBAL_STORE: dict[str, ConfigStore] = {}
_GLOBAL_STORE_LOCK = threading.Lock()
_GLOBAL_STORE_KEY = 'global'
GLOBAL_STORE_LABEL = '__global__'


def get_global_config_store(*, mode: Literal['live', 'pinned'] = 'live') -> ConfigStore:
    """The process's singleton :class:`ConfigStore` for the deployment-wide registry.

    Unlike :func:`~src.services.config_store.store.get_config_store` (keyed by, and unusable
    without, the *bound* project), this never consults the project context at all: it requires
    no project binding, and calling it while a project happens to be bound has no effect on
    which store it returns.
    """
    with _GLOBAL_STORE_LOCK:
        store = _GLOBAL_STORE.get(_GLOBAL_STORE_KEY)
        if store is None:
            store = ConfigStore(mode=mode, label=GLOBAL_STORE_LABEL, is_global=True)
            _GLOBAL_STORE[_GLOBAL_STORE_KEY] = store
        return store


def reset_global_config_store() -> None:
    """Test-only: drop the cached global store so a test starts from a clean snapshot."""
    with _GLOBAL_STORE_LOCK:
        _GLOBAL_STORE.clear()


__all__ = ['GLOBAL_STORE_LABEL', 'get_global_config_store', 'reset_global_config_store']
