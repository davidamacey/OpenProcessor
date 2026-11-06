"""Persistence of the fitted UMAP reducer (visualization only).

The fitted reducer is a blob behind the storage seam, key ``projects/<slug>/umap/<state_id>.bin``.
Slots are per backend because sklearn and cuML pickle incompatible class instances: ``current``
is the sklearn slot, ``current_cuml`` the GPU one.
"""

from __future__ import annotations

import io
from typing import Any

from src.clients.storage.roots import RootedStore, build_rooted_store
from src.config.project_context import current_project
from src.config.storage import storage_root_paths
from src.core.logging import get_logger
from src.domain.storage_keys import make_key


logger = get_logger(__name__)


def state_id_for(backend: str) -> str:
    """The reducer slot of a backend (a worker flipping CPU and GPU must not unpickle a sklearn
    UMAP as a cuML one, or the reverse)."""
    return 'current_cuml' if backend == 'gpu' else 'current'


def reducer_key(state_id: str) -> str:
    return make_key('projects', current_project().record.slug, 'umap', f'{state_id}.bin')


def _store() -> RootedStore:
    return build_rooted_store(storage_root_paths())


def serialize_reducer(reducer: Any) -> bytes:
    import joblib

    buf = io.BytesIO()
    joblib.dump(reducer, buf)
    return buf.getvalue()


def _deserialize_reducer(blob: bytes) -> Any:
    import joblib

    return joblib.load(io.BytesIO(blob))


def save_reducer(reducer: Any, *, state_id: str) -> str:
    """Write the reducer blob and return its storage key."""
    key = reducer_key(state_id)
    _store().write_atomic(key, serialize_reducer(reducer))
    return key


def reducer_stored(*, state_id: str) -> str | None:
    """The storage key of the reducer when one is stored, else ``None``."""
    key = reducer_key(state_id)
    return key if _store().exists(key) else None


def load_reducer(*, state_id: str) -> Any | None:
    """The stored reducer, or ``None`` when absent or unreadable (a stale or backend-incompatible
    pickle is logged and refit; last week's reducer must not fail a run)."""
    store = _store()
    key = reducer_key(state_id)
    if not store.exists(key):
        return None
    try:
        with store.open_read(key) as handle:
            return _deserialize_reducer(handle.read())
    except Exception as exc:
        logger.warning('curation_umap_state_load_failed', key=key, error=str(exc))
        return None
