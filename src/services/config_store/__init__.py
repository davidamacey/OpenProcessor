"""The config store: prompt packs, region profiles, open-vocabulary sets, ingest profiles and VLM
endpoints as versioned documents in PostgreSQL, with the activations per project, a config epoch
every write bumps, a per-process snapshot and hot reload."""

from __future__ import annotations

from src.services.config_store.documents import (
    ActiveConflictError,
    RevisionConflictError,
    activate,
    delete_config,
    save_config,
)
from src.services.config_store.global_store import (
    get_global_config_store,
    reset_global_config_store,
)
from src.services.config_store.snapshot import AxisRef, ConfigSnapshot, StoredConfig
from src.services.config_store.store import (
    ConfigStore,
    ConfigStoreUnavailableError,
    activate_axis,
    config_store_for,
    get_config_store,
    reset_config_stores,
)
from src.services.config_store.store_poll import (
    shutdown_config_store_poll,
    startup_bootstrap_config_store_safe,
)


__all__ = [
    'ActiveConflictError',
    'AxisRef',
    'ConfigSnapshot',
    'ConfigStore',
    'ConfigStoreUnavailableError',
    'RevisionConflictError',
    'StoredConfig',
    'activate',
    'activate_axis',
    'config_store_for',
    'delete_config',
    'get_config_store',
    'get_global_config_store',
    'reset_config_stores',
    'reset_global_config_store',
    'save_config',
    'shutdown_config_store_poll',
    'startup_bootstrap_config_store_safe',
]
