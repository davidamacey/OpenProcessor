"""Per-cluster AHC refinement of item clusters (and the shared refine core).

Public surface: :func:`refine_cluster` (``POST
/curation/clusters/refine/{cluster_id}``) and :func:`refine_members`, the
AHC core shared with region-box clusters.

Why complete + cosine + threshold:

- **Complete linkage** uses the *maximum* pairwise distance when merging,
  so a sub-cluster only grows if every member stays within
  ``distance_threshold`` of every other member.
- **Cosine distance** matches the PE / classifier training objective.
- **distance_threshold=0.25** (~75 % cosine similarity) — shared between
  refine and the AHC residual fallback so both surfaces are calibrated
  identically.

Skip-rules: > MAX_REFINE_MEMBERS (default 8000) members (skip+warn);
< MIN_REFINE_MEMBERS (4) members (skip+info).
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.config.curation import items_index
from src.core.logging import get_logger
from src.services.curation.clustering.cluster_write_guard import _log_bulk_write_errors
from src.services.curation.clustering.methods.ahc import (
    AHC_DISTANCE_THRESHOLD,
    AHC_LINKAGE,
    AHC_METRIC,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


# Refinement thresholds.
# > this — refinement is skipped. The refine path runs sklearn AHC with NO
# connectivity graph, so it builds a FULL pairwise distance matrix: ~8*n^2
# bytes (float64). Rough transient RAM in the api process:
#   2000 -> ~32 MB    5000 -> ~200 MB    8000 -> ~512 MB    12000 -> ~1.15 GB
# Plus ~4 KB/member to fetch embeddings. It's fast (AHC fit is seconds even
# at 8k) and off-loaded to a worker thread, so the binding constraint is RAM,
# not latency. Raise via OP_MAX_REFINE_MEMBERS as far as the container allows.
MAX_REFINE_MEMBERS = int(os.getenv('OP_MAX_REFINE_MEMBERS', '8000'))
# < this — skip (AHC needs at least a few points). The floor used to be 50 on
# the theory that small clusters don't benefit, but operators also use refine
# purely to *organize* a small bucket into like-kind sub-groups for faster
# select-and-label, so the floor is low — just enough to keep AHC well-defined.
MIN_REFINE_MEMBERS = 4


def subcluster_label(idx: int) -> str:
    """Convert a sub-cluster index ``0,1,2,...`` into a label suffix ``a,b,c,...,aa,ab,...``."""
    if idx < 0:
        raise ValueError('subcluster index must be >= 0')
    out = ''
    n = idx
    while True:
        out = chr(ord('a') + (n % 26)) + out
        n = n // 26 - 1
        if n < 0:
            break
    return out


async def _fetch_cluster_members(
    client: AsyncOpenSearch,
    cluster_id: int,
    *,
    page_size: int = 1000,
    index: str | None = None,
) -> list[dict[str, Any]]:
    """Pull every item with ``cluster_id == cluster_id`` from ``index``.

    Reads the item residual embedding (``pe_embedding``), 1024x4B ~ 4KB
    each, so MAX_REFINE_MEMBERS (default 8000) members ~ 32MB -- safe to
    load into RAM. The embedding is normalized to the ``'embedding'`` key
    so :func:`refine_members` stays field-name-agnostic.
    """
    from src.services.curation.clustering.embedding_reduce import RESIDUAL_EMBEDDING_FIELD

    if index is None:
        index = items_index()
    embedding_field = RESIDUAL_EMBEDDING_FIELD

    members: list[dict[str, Any]] = []
    body = {
        'size': page_size,
        'query': {'term': {'cluster_id': cluster_id}},
        '_source': ['crop_id', 'class_name', 'class_id', 'class_validated', embedding_field],
    }
    resp = await client.search(index=index, body=body, scroll='2m')
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    while hits:
        for h in hits:
            src = h.get('_source') or {}
            emb = src.get(embedding_field)
            if emb is None:
                continue
            members.append({**src, 'embedding': emb, '_id': h['_id']})
        resp = await client.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']

    if scroll_id:
        try:
            await client.clear_scroll(scroll_id=scroll_id)
        except Exception as e:
            logger.warning('curation_clear_scroll_failed', error=str(e))
    return members


_SUBID_UPDATE_CHUNK = 1000


async def _bulk_update_subids(
    client: AsyncOpenSearch,
    updates: list[tuple[str, str]],
    *,
    index: str | None = None,
    expected_cluster_id: int | None = None,
    chunk_size: int = _SUBID_UPDATE_CHUNK,
) -> int:
    """Bulk-update ``cluster_subid`` on the supplied (doc_id, subid) pairs.

    When ``expected_cluster_id`` is given (refine's caller always passes it
    -- the cluster being refined), the write is a guarded painless script
    that noops if the doc's ``cluster_id`` no longer equals it. Refine
    snapshots members, fits AHC (can take seconds on a large cluster), then
    writes; a doc that moved to a different cluster in that window (a human
    relabel, a move endpoint call, another clustering job) must not have
    refine's now-stale sub-cluster numbering stamped onto it.

    Chunks into batches of ``chunk_size`` (<=1000) bulk actions with
    ``refresh=False`` per chunk, then issues one explicit index refresh at
    the end -- avoids refreshing the index once per chunk on a large refine.
    """
    if index is None:
        index = items_index()
    if not updates:
        return 0
    now = datetime.now(UTC).isoformat()
    for start in range(0, len(updates), chunk_size):
        chunk = updates[start : start + chunk_size]
        body: list[dict[str, Any]] = []
        for doc_id, subid in chunk:
            body.append({'update': {'_index': index, '_id': doc_id}})
            if expected_cluster_id is None:
                body.append({'doc': {'cluster_subid': subid, 'updated_at': now}})
            else:
                body.append(
                    {
                        'script': {
                            'lang': 'painless',
                            'params': {'cid': expected_cluster_id, 'subid': subid, 'now': now},
                            'source': (
                                "if (ctx._source['cluster_id'] != params.cid)"
                                " { ctx.op = 'noop'; return; }"
                                " ctx._source['cluster_subid'] = params.subid;"
                                ' ctx._source.updated_at = params.now;'
                            ),
                        }
                    }
                )
        resp = await client.bulk(body=body, refresh=False)
        if resp.get('errors'):
            _log_bulk_write_errors('bulk_update_subids', resp)
    try:
        await client.indices.refresh(index=index)
    except Exception as exc:
        logger.warning('bulk_update_subids_refresh_failed', error=str(exc))
    return len(updates)


def _compute_purity(class_names: list[str | None]) -> float:
    """Largest-class share among labelled members (None excluded). 0.0 if all None."""
    labelled = [c for c in class_names if c]
    if not labelled:
        return 0.0
    counts = Counter(labelled)
    top = counts.most_common(1)[0][1]
    return top / len(labelled)


async def refine_members(
    cluster_id: int,
    *,
    count_members: Callable[[], Awaitable[int]],
    fetch_members: Callable[[], Awaitable[list[dict[str, Any]]]],
    write_subids: Callable[[list[tuple[Any, str]]], Awaitable[int]],
    unit: Literal['items', 'boxes'],
    distance_threshold: float = AHC_DISTANCE_THRESHOLD,
    max_members: int = MAX_REFINE_MEMBERS,
) -> dict[str, Any]:
    """The AHC refine core shared by item clusters and region-box clusters.

    ``unit`` is what a member is, and names the response counts
    (``n_<unit>`` members, ``n_<unit>_updated`` members whose stored
    sub-id ``write_subids`` reports changed).

    ``fetch_members`` returns ``{'_id': <opaque key>, 'embedding': [...],
    'class_name': ...}`` dicts; ``write_subids`` receives ``(_id, subid)``
    pairs. Steps:

    1. ``count_members`` first -- a cluster far past ``max_members`` never
       pays for fetching every embedding just to learn it is too large.
    2. Skip if < MIN_REFINE_MEMBERS (4) members (too small) or
       > ``max_members`` (default 8000) members (too expensive).
    3. ``AgglomerativeClustering(linkage='complete', distance_threshold=0.25,
       metric='cosine')`` over the embeddings.
    4. ``write_subids`` the ``"47a"``, ``"47b"`` ... labels; every current
       member gets a fresh one, so re-running overwrites a previous partition.
    5. Purity (largest-class share among labelled members) in the summary.

    Returns ``{cluster_id, n_<unit>, n_subclusters, purity, action, ...}``.
    """
    log = logger.bind(cluster_id=cluster_id)
    log.info('curation_refine_cluster_start')
    n_key = f'n_{unit}'

    precount = await count_members()
    if precount > max_members:
        log.warning(
            'curation_refine_cluster_skipped_too_large_precount',
            n_members=precount,
            max_allowed=max_members,
        )
        return {
            'cluster_id': cluster_id,
            n_key: precount,
            'n_subclusters': 0,
            # Purity isn't computed here -- that would need the same full
            # fetch this precount check exists to avoid paying for.
            'purity': None,
            'action': 'skipped_too_large',
            'reason': (
                f'> {max_members} members ({precount} counted); AHC builds a full '
                '~8*n^2-byte pairwise matrix -- raise OP_MAX_REFINE_MEMBERS / '
                'max_members if RAM allows'
            ),
        }

    members = await fetch_members()
    n_members = len(members)

    if n_members < MIN_REFINE_MEMBERS:
        log.info(
            'curation_refine_cluster_skipped_too_small',
            n_members=n_members,
            min_required=MIN_REFINE_MEMBERS,
        )
        return {
            'cluster_id': cluster_id,
            n_key: n_members,
            'n_subclusters': 0,
            'purity': _compute_purity([m.get('class_name') for m in members]),
            'action': 'skipped_too_small',
            'reason': f'< {MIN_REFINE_MEMBERS} members; AHC offers no benefit',
        }

    if n_members > max_members:
        log.warning(
            'curation_refine_cluster_skipped_too_large',
            n_members=n_members,
            max_allowed=max_members,
        )
        return {
            'cluster_id': cluster_id,
            n_key: n_members,
            'n_subclusters': 0,
            'purity': _compute_purity([m.get('class_name') for m in members]),
            'action': 'skipped_too_large',
            'reason': (
                f'> {max_members} members; AHC builds a full ~8*n^2-byte pairwise '
                'matrix -- raise OP_MAX_REFINE_MEMBERS / max_members if RAM allows'
            ),
        }

    try:
        embeddings = np.asarray([m['embedding'] for m in members], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as e:
        log.error('curation_refine_cluster_embedding_load_failed', error=str(e))
        raise

    # sklearn import is local to keep startup fast and avoid a hard dep for
    # callers that never touch clustering.
    from sklearn.cluster import AgglomerativeClustering

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage=AHC_LINKAGE,
        metric=AHC_METRIC,
    )
    # Off-load to a worker thread so a large cluster (close to
    # MAX_REFINE_MEMBERS) doesn't block the FastAPI event loop -- refine runs
    # in the api process, so a sync fit here would starve every other
    # request.
    sub_labels = await asyncio.to_thread(clusterer.fit_predict, embeddings)
    n_subclusters = int(sub_labels.max() + 1) if len(sub_labels) > 0 else 0

    updates = [
        (member['_id'], f'{cluster_id}{subcluster_label(int(sub_idx))}')
        for member, sub_idx in zip(members, sub_labels, strict=True)
    ]
    n_updated = await write_subids(updates)

    # Per-sub-cluster purity, then weighted-mean as the cluster summary.
    sub_groups: dict[int, list[str | None]] = {}
    for member, sub_idx in zip(members, sub_labels, strict=True):
        sub_groups.setdefault(int(sub_idx), []).append(member.get('class_name'))
    weighted_purity = (
        sum(_compute_purity(names) * len(names) for names in sub_groups.values()) / n_members
    )

    summary: dict[str, Any] = {
        'cluster_id': cluster_id,
        n_key: n_members,
        'n_subclusters': n_subclusters,
        'purity': _compute_purity([m.get('class_name') for m in members]),
        'subcluster_weighted_purity': weighted_purity,
        f'n_{unit}_updated': n_updated,
        'distance_threshold': distance_threshold,
        'linkage': AHC_LINKAGE,
        'metric': AHC_METRIC,
        'action': 'refined',
    }
    log.info('curation_refine_cluster_done', **summary)
    return summary


async def refine_cluster(
    client: AsyncOpenSearch,
    cluster_id: int,
    *,
    distance_threshold: float = AHC_DISTANCE_THRESHOLD,
    max_members: int = MAX_REFINE_MEMBERS,
) -> dict[str, Any]:
    """Per-cluster AHC refinement of an *item* cluster (``pe_embedding`` /
    ``cluster_id`` / ``cluster_subid``); see :func:`refine_members`. Region
    boxes refine through ``region_box_clustering.refine_region_cluster``."""
    index = items_index()

    async def count_members() -> int:
        resp = await client.count(index=index, body={'query': {'term': {'cluster_id': cluster_id}}})
        return int((resp or {}).get('count', 0))

    async def write_subids(updates: list[tuple[Any, str]]) -> int:
        return await _bulk_update_subids(
            client, updates, index=index, expected_cluster_id=cluster_id
        )

    return await refine_members(
        cluster_id,
        count_members=count_members,
        fetch_members=lambda: _fetch_cluster_members(client, cluster_id, index=index),
        write_subids=write_subids,
        unit='items',
        distance_threshold=distance_threshold,
        max_members=max_members,
    )
