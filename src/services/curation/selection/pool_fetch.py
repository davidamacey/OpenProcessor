"""Shared embedding-pool fetch for the selection overlay
(curation-strategy plan §3.4/§7 Phase 4).

Scroll-based ``ids + pe_embedding`` fetch for an arbitrary OpenSearch query,
capped at a caller-supplied ``cap`` — same scroll-then-break shape as
:func:`src.services.curation.clustering.outliers.compute_outlier_order`'s
member fetch, generalized to an arbitrary query clause instead of one
hardcoded to a single cluster. Lives in ``selection/`` (not the
``legacy_select.py`` router) so both the router's synchronous path and
:mod:`selection.job`'s backgrounded path import the same code — a router
module must not be a dependency of a service module (the reverse is the
normal direction everywhere else in this package).

Pure read: never touches ``cluster_id``/``cluster_subid``/``cluster_distance``
or writes anything back to OpenSearch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)

EMBEDDING_FIELD = 'pe_embedding'
_SCROLL_PAGE = 2000


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Defensive re-normalization — ``pe_embedding`` is stored unit-norm,
    but ``compute_uniqueness``/``compute_outlier_order`` both re-normalize
    before using dot products as cosine similarity, so this mirrors that
    same "be safe" convention rather than trusting storage invariants."""
    xf = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(xf, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return xf / norms


async def fetch_pool_embeddings(
    client: AsyncOpenSearch,
    index: str,
    query: dict[str, Any],
    *,
    cap: int,
    embedding_field: str = EMBEDDING_FIELD,
) -> tuple[list[str], np.ndarray, bool]:
    """Scroll ``query``'s matches, collecting ``crop_id`` + ``embedding_field``.

    Breaks out (without finishing the scroll) as soon as more than ``cap``
    docs have been seen, mirroring ``compute_outlier_order``'s
    too-large-for-interactive-request behavior. Returns
    ``(ids, embeddings, truncated)`` — when ``truncated`` is ``True``,
    ``embeddings`` is an empty ``(0, 0)`` array (the caller should not use
    it; a truncated pool means "give up on this cap", not "here's a
    partial answer").
    """
    # F-16: count first — a pool far past `cap` should never pay for a
    # scroll (even a break-early one) just to discover it's too large.
    count_resp = await client.count(index=index, body={'query': query})
    if int((count_resp or {}).get('count', 0)) > cap:
        return [], np.zeros((0, 0), dtype=np.float32), True

    ids: list[str] = []
    vecs: list[list[float]] = []
    body = {'size': _SCROLL_PAGE, 'query': query, '_source': [embedding_field]}
    resp = await client.search(index=index, body=body, scroll='2m')
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    truncated = False
    while hits:
        for h in hits:
            emb = (h.get('_source') or {}).get(embedding_field)
            if emb is not None:
                ids.append(h['_id'])
                vecs.append(emb)
        if len(ids) > cap:
            truncated = True
            break
        resp = await client.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']
    if scroll_id:
        try:
            await client.clear_scroll(scroll_id=scroll_id)
        except Exception as exc:
            logger.warning('curation_select_clear_scroll_failed', error=str(exc))

    if truncated:
        return ids[:cap], np.zeros((0, 0), dtype=np.float32), True
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32), False
    return ids, np.asarray(vecs, dtype=np.float32), False


__all__ = ['EMBEDDING_FIELD', 'fetch_pool_embeddings', 'l2_normalize']
