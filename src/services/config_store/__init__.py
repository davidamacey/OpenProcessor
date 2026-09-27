"""The config store (W2): OpenSearch-backed prompt packs and region
profiles with a global revision counter, per-process snapshot and hot
reload -- no CRUD routes live here yet (W3/W4). See
``docs/design/openprocessor_internal/any_domain_plan.md`` §3/§4/§9 W2.
"""

from __future__ import annotations

from src.services.config_store.index import (
    ActiveConflictError,
    RevisionConflictError,
    activate,
    activation_doc_id,
    bump_config_revision,
    config_doc_id,
    delete_config,
    get_activation,
    get_config_revision,
    get_runtime_docs,
    rollback,
    save_config,
    upsert_runtime_doc,
)
from src.services.config_store.store import (
    AxisRef,
    ConfigSnapshot,
    ConfigStore,
    StoredConfig,
    activate_axis,
    get_config_store,
    reset_config_stores,
    shutdown_config_store_poll,
    startup_bootstrap_config_store_safe,
)


__all__ = [
    'ActiveConflictError',
    'AxisRef',
    'ConfigSnapshot',
    'ConfigStore',
    'RevisionConflictError',
    'StoredConfig',
    'activate',
    'activate_axis',
    'activation_doc_id',
    'bump_config_revision',
    'config_doc_id',
    'delete_config',
    'get_activation',
    'get_config_revision',
    'get_config_store',
    'get_runtime_docs',
    'reset_config_stores',
    'rollback',
    'save_config',
    'shutdown_config_store_poll',
    'startup_bootstrap_config_store_safe',
    'upsert_runtime_doc',
]
