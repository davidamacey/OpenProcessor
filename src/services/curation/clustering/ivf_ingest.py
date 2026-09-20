"""In-process IVF centroid cache + ingest quality gate.

Ported from the private reference ingest service's residual-clustering
option (see ``docs/design/curation_design_rationale.md`` §2.1 for the
citation convention; reference lines ``:165-229``). Two independent
things live here, both process-cached and keyed on the persisted
centroids file's mtime so a worker retrain propagates to the ingest
path without a process restart:

1. :func:`get_ivf_ingest_store` — a cached, loaded
   :class:`~src.services.curation.clustering.methods.ivf_store.IVFCentroidStore`
   (or ``None`` if no centroids are trained yet). The curation ingest
   service uses this to assign a residual (unconfident) item a
   candidate ``cluster_id`` immediately, instead of waiting for the
   next batch recluster.
2. :func:`ingest_passes_gate` — the primary-subject quality gate
   (``max_rank`` / ``min_blur_ratio``) trained alongside the centroids
   and persisted next to them (``IVFCentroidStore.load_gate``). A crop
   failing the gate is parked (``PARKED_CLUSTER_ID``) rather than
   assigned to a real candidate cluster, keeping cluster cards free of
   tiny/blurry background objects until a looser recluster reconsiders it.
"""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.services.curation.clustering.methods.ivf_store import CENTROIDS_PATH, IVFCentroidStore


logger = get_logger(__name__)


# Process-cached centroid store for ingest assignment, keyed on the
# centroids file mtime so a worker retrain propagates here without a
# process restart. Dict cache avoids a module-level `global` rebind.
_ivf_ingest_cache: dict[str, Any] = {'mtime': None, 'store': None, 'gate': {}}


def get_ivf_ingest_store() -> IVFCentroidStore | None:
    """Return a loaded :class:`IVFCentroidStore`, or ``None`` if unavailable.

    The FAISS centroid index is cached in-process and reused across
    crops. The centroids file's mtime is stat'd on each call (cheap)
    and the store is reloaded only when it changes — so a periodic
    worker retrain is picked up automatically. A failed/missing load
    caches ``None`` for that mtime so a broken file isn't repeatedly
    retried within the same generation.
    """
    try:
        mtime = CENTROIDS_PATH.stat().st_mtime
    except OSError:
        # No centroids persisted yet (fresh deploy). Cache the miss.
        _ivf_ingest_cache['mtime'] = None
        _ivf_ingest_cache['store'] = None
        _ivf_ingest_cache['gate'] = {}
        return None

    if _ivf_ingest_cache['mtime'] == mtime:
        return _ivf_ingest_cache['store']

    # mtime changed (or first load) — (re)load the centroids + gate
    # policy together, since a gated full recluster always rewrites the
    # centroids alongside the gate it was trained under.
    _ivf_ingest_cache['mtime'] = mtime
    try:
        store = IVFCentroidStore()
        loaded = store.load()
        _ivf_ingest_cache['store'] = store if loaded else None
        _ivf_ingest_cache['gate'] = store.load_gate() if loaded else {}
    except Exception as exc:
        logger.warning('ivf_ingest_store_load_failed', error=str(exc))
        _ivf_ingest_cache['store'] = None
        _ivf_ingest_cache['gate'] = {}
    return _ivf_ingest_cache['store']


def ingest_passes_gate(rank: int | None, blur_ratio_val: float | None) -> bool:
    """Whether a crop clears the active ingest clustering gate.

    Reads the gate cached by :func:`get_ivf_ingest_store` (call that
    first so the cache reflects the current centroids generation).
    Strict (matches the clustering partition): a missing field fails
    the gate. Returns ``True`` when no gate is active.
    """
    gate = _ivf_ingest_cache.get('gate') or {}
    max_rank = gate.get('max_rank')
    min_blur_ratio = gate.get('min_blur_ratio')
    fails_rank = max_rank is not None and (rank is None or rank > max_rank)
    fails_blur = min_blur_ratio is not None and (
        blur_ratio_val is None or blur_ratio_val < min_blur_ratio
    )
    return not (fails_rank or fails_blur)


def reset_ivf_ingest_cache() -> None:
    """Clear the in-process cache. Test-only convenience."""
    _ivf_ingest_cache['mtime'] = None
    _ivf_ingest_cache['store'] = None
    _ivf_ingest_cache['gate'] = {}


__all__ = [
    'get_ivf_ingest_store',
    'ingest_passes_gate',
    'reset_ivf_ingest_cache',
]
