"""Shared embedding-pool fetch for the selection overlay.

``ids + pe_embedding`` fetch (``read_vectors``) for an arbitrary OpenSearch query,
capped at a caller-supplied ``cap`` — same scroll-then-break shape as
:func:`src.services.curation.clustering.outliers.compute_outlier_order`'s
member fetch, generalized to an arbitrary query clause instead of one
hardcoded to a single cluster. Lives in ``selection/`` (not the
select router) so both the router's synchronous path and
:mod:`selection.job`'s backgrounded path import the same code — a router
module must not be a dependency of a service module (the reverse is the
normal direction everywhere else in this package).

Pure read: never touches ``cluster_id``/``cluster_subid``/``cluster_distance``
or writes anything back to OpenSearch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.services.curation.clustering.vector_read import read_vectors


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Defensive re-normalization — the stored vector is unit-norm,
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
    embedding_field: str,
) -> tuple[list[str], np.ndarray, bool]:
    """Read ``query``'s matches, collecting ``crop_id`` + ``embedding_field``.

    Stops reading as soon as more than ``cap``
    docs have been seen, mirroring ``compute_outlier_order``'s
    too-large-for-interactive-request behavior. Returns
    ``(ids, embeddings, truncated)`` — when ``truncated`` is ``True``,
    ``embeddings`` is an empty ``(0, 0)`` array (the caller should not use
    it; a truncated pool means "give up on this cap", not "here's a
    partial answer").
    """
    # Count first — a pool far past `cap` should never pay for a
    # scroll (even a break-early one) just to discover it's too large.
    count_resp = await client.count(index=index, body={'query': query})
    if int((count_resp or {}).get('count', 0)) > cap:
        return [], np.zeros((0, 0), dtype=np.float32), True

    read = await read_vectors(client, index=index, field=embedding_field, query=query, max_rows=cap)
    return read.ids, read.matrix, read.truncated


__all__ = ['fetch_pool_embeddings', 'l2_normalize']
