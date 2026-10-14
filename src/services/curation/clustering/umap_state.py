"""Persistence of the fitted UMAP reducer (visualization only).

The fitted reducer is cached in two places: a local joblib file under the
bound project's state dir and the OpenSearch ``op_umap_state`` index.
Cache slots are per-backend because sklearn and cuML pickle incompatible
class instances.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.config.curation import umap_state_index
from src.core.logging import get_logger
from src.services.curation.clustering.pool_size import UMAP_METRIC, UMAP_N_COMPONENTS


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


# The reducer cache is project data: it lives under the bound project's
# state dir (``project_state_dir``; ``state_dir`` itself for ``default``),
# resolved per call.
def umap_state_joblib_path() -> str:
    return str(Path(get_curation_config().project_state_dir) / 'umap_state.joblib')


def umap_state_joblib_path_cuml() -> str:
    return str(Path(get_curation_config().project_state_dir) / 'umap_state_cuml.joblib')


def _state_paths_for(backend: str) -> tuple[str, str]:
    """Return ``(joblib_path, opensearch_state_id)`` for the given backend.

    Per-backend cache slots so a worker flipping CPU<->GPU doesn't try
    to unpickle a sklearn UMAP class as a cuML UMAP class (or vice
    versa). The ``current`` OpenSearch state id is the legacy sklearn
    slot; the GPU slot is ``current_cuml``.
    """
    if backend == 'gpu':
        return umap_state_joblib_path_cuml(), 'current_cuml'
    return umap_state_joblib_path(), 'current'


def _serialize_reducer(reducer: Any) -> bytes:
    import joblib

    buf = io.BytesIO()
    joblib.dump(reducer, buf)
    return buf.getvalue()


def _deserialize_reducer(blob: bytes) -> Any:
    import joblib

    return joblib.load(io.BytesIO(blob))


def _save_umap_state_to_disk(reducer: Any, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_serialize_reducer(reducer))


def _load_umap_state_from_disk(path: str) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return _deserialize_reducer(p.read_bytes())
    except Exception as exc:
        # Stale or backend-incompatible pickle — log + refit. Don't
        # crash the run because last week's reducer can't unpickle in
        # the new container.
        logger.warning('curation_umap_state_disk_load_failed', path=path, error=str(exc))
        return None


# OpenSearch's default HTTP request limit is 100 MB. cuML's pickled
# UMAP retains the training data on the GPU (cuml#5818), which inflates
# the joblib blob to ~120 MB+ at 90k x 1024. Skip the OpenSearch
# fallback when the blob is over this threshold — the disk-cached copy
# is still written, and a refit on a fresh container is acceptable.
_OPENSEARCH_PERSIST_MAX_BYTES = 60 * 1024 * 1024


async def _save_umap_state_to_opensearch(
    client: AsyncOpenSearch, reducer: Any, *, state_id: str
) -> None:
    blob = _serialize_reducer(reducer)
    if len(blob) > _OPENSEARCH_PERSIST_MAX_BYTES:
        logger.info(
            'curation_umap_state_skip_opensearch_persist',
            reason='blob_too_large',
            blob_bytes=len(blob),
            limit_bytes=_OPENSEARCH_PERSIST_MAX_BYTES,
            state_id=state_id,
        )
        return
    body = {
        'state_id': state_id,
        'reducer_b64': base64.b64encode(blob).decode('ascii'),
        'n_components': UMAP_N_COMPONENTS,
        'metric': UMAP_METRIC,
    }
    await client.index(index=umap_state_index(), id=state_id, body=body, refresh=False)


async def _load_umap_state_from_opensearch(client: AsyncOpenSearch, *, state_id: str) -> Any | None:
    try:
        resp = await client.get(index=umap_state_index(), id=state_id)
    except Exception as exc:
        logger.debug('curation_umap_state_not_found', state_id=state_id, error=str(exc))
        return None
    src = resp.get('_source') or {}
    blob_b64 = src.get('reducer_b64')
    if not blob_b64:
        return None
    try:
        return _deserialize_reducer(base64.b64decode(blob_b64))
    except Exception as exc:
        logger.warning('curation_umap_state_os_load_failed', state_id=state_id, error=str(exc))
        return None
