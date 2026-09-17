"""On-the-fly outlier ranking for a cluster, by distance from its centroid.

Vehicle CLASS clusters (cluster_id == class_id) are populated from v6/human
labels, not from a FAISS centroid, so they carry no ``cluster_distance``.
To let operators cherry-pick the worst offenders (a car sitting in the
"pickup" cluster, a mislabel, a junk crop), we compute the class centroid as
the mean of the members' ``pe_embedding`` and rank each member by cosine
distance from it — farthest first. Same idea a folder-sorting utility
used to float outliers to the top of a review folder.

The expensive part is the OpenSearch scroll of the members' embeddings, so
the ranked order is cached per (index, query) for a TTL. The centroid is
stable under in-cluster edits (validating a label doesn't move it), and a
cheap member-count check invalidates the cache when membership changes
(crops moved in/out), so a stale order never outlives a real change for long.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)

OUTLIER_EMBEDDING_FIELD = 'pe_embedding'
# Above this many members the pairwise scroll + sort isn't worth it for an
# interactive request — callers fall back to the default sort.
_MAX_MEMBERS = int(os.getenv('OP_OUTLIER_MAX_MEMBERS', '20000'))
_TTL_S = float(os.getenv('OP_OUTLIER_CACHE_TTL_S', '600'))
_SCROLL_PAGE = 2000

# cache_key -> {'order': list[str], 'count': int, 'at': float}
_CACHE: dict[str, dict[str, Any]] = {}


def make_cache_key(index: str, query: dict[str, Any], embedding_field: str) -> str:
    """Stable key for a (index, query, field) triple."""
    return f'{index}|{embedding_field}|{json.dumps(query, sort_keys=True)}'


async def compute_outlier_order(
    client: AsyncOpenSearch,
    index: str,
    query: dict[str, Any],
    *,
    embedding_field: str = OUTLIER_EMBEDDING_FIELD,
    current_count: int | None = None,
) -> list[str] | None:
    """Return member doc ``_id``s ordered by descending centroid distance.

    Cached per (index, query, field). ``current_count`` (the live count for
    the same query) invalidates a cached order when membership changed.
    Returns ``None`` when the cluster exceeds ``_MAX_MEMBERS`` (caller should
    fall back to its default sort).
    """
    key = make_cache_key(index, query, embedding_field)
    now = time.monotonic()
    cached = _CACHE.get(key)
    if (
        cached is not None
        and (now - cached['at']) < _TTL_S
        and (current_count is None or cached['count'] == current_count)
    ):
        return cached['order']  # type: ignore[no-any-return]

    ids: list[str] = []
    vecs: list[list[float]] = []
    body = {'size': _SCROLL_PAGE, 'query': query, '_source': [embedding_field]}
    resp = await client.search(index=index, body=body, scroll='2m')
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    too_large = False
    while hits:
        for h in hits:
            emb = (h.get('_source') or {}).get(embedding_field)
            if emb is not None:
                ids.append(h['_id'])
                vecs.append(emb)
        if len(ids) > _MAX_MEMBERS:
            too_large = True
            break
        resp = await client.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']
    if scroll_id:
        try:
            await client.clear_scroll(scroll_id=scroll_id)
        except Exception as exc:
            logger.warning('legacy_outlier_clear_scroll_failed', error=str(exc))

    if too_large:
        logger.info('legacy_outlier_skip_too_large', index=index, n_seen=len(ids))
        return None
    if not ids:
        return []

    x = np.asarray(vecs, dtype=np.float32)
    # Unit-normalize members (pe_embedding is already L2-normed, but be safe),
    # build the centroid, normalize it, then cosine distance = 1 - cos.
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    centroid = xn.mean(axis=0)
    cnorm = float(np.linalg.norm(centroid))
    if cnorm > 0:
        centroid = centroid / cnorm
    dist = 1.0 - (xn @ centroid)
    order = [ids[i] for i in np.argsort(-dist)]  # farthest (most atypical) first

    _CACHE[key] = {'order': order, 'count': len(order), 'at': now}
    return order


__all__ = ['OUTLIER_EMBEDDING_FIELD', 'compute_outlier_order', 'make_cache_key']
