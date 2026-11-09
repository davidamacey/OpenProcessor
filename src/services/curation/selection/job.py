"""The queued ``selection`` job behind pool-scale ``POST /curation/select/diverse`` requests.

The select router answers inline when the scoped pool is small enough (see its module
docstring); above the documented sync-ops budget it queues this job instead. A Celery worker runs
:func:`run_selection_job`: k-center-greedy at pool scale (n~128k, k~1000) is 1-2 minutes of CPU,
too slow for a request.

Read-only with respect to crop documents: the job only reads embeddings and returns the selected
ids as its result; it never issues an OpenSearch ``update`` or ``bulk`` write.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.curation.latest_job import common_state
from src.services.jobs.service import start_job


if TYPE_CHECKING:
    import uuid

    import numpy as np
    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repos.jobs import JobRow
    from src.schemas.jobs import JobView
    from src.tasks.runtime.context import JobContext

logger = get_logger(__name__)

KIND = 'selection'


async def start(
    session: AsyncSession,
    project: uuid.UUID,
    *,
    index: str,
    query: dict[str, Any],
    k: int,
    seed_crop_id: str | None,
    scope: dict[str, Any],
    max_n: int,
    embedding_field: str,
) -> JobRow:
    """Queue the selection (409 ``job_busy`` while one is active)."""
    started = await start_job(
        session,
        KIND,
        project,
        params={
            'index': index,
            'query': query,
            'k': k,
            'seed_crop_id': seed_crop_id,
            'scope': scope,
            'max_n': max_n,
            'embedding_field': embedding_field,
        },
        requested_by='api',
    )
    return started.job


def state_dict(view: JobView | None) -> dict[str, Any]:
    """The ``/select/status`` wire of a job row."""
    params = (view.params if view else None) or {}
    progress = (view.progress if view else None) or {}
    result = (view.result or {}).get('selection') if view else None
    return {
        **common_state(view),
        'k': int(params.get('k') or 0),
        'scope': dict(params.get('scope') or {}),
        'n_pool': int(progress.get('total') or 0),
        'result': result,
    }


async def run_selection_job(ctx: JobContext, opensearch: AsyncOpenSearch) -> dict[str, Any]:
    """Actual compute body: one pool fetch (capped only by the generous job-path ``max_n``), one
    ``k_center_greedy`` call, done. A cancel is checked once before the (uninterruptible) greedy
    call: its tight numpy loop has no natural point to look inside."""
    from src.services.curation.selection import k_center_greedy
    from src.services.curation.selection.pool_fetch import fetch_pool_embeddings, l2_normalize

    params = ctx.params
    if ctx.cancelled():
        return {'cancelled': True}
    ids, embeddings, _truncated = await fetch_pool_embeddings(
        opensearch,
        params['index'],
        params['query'],
        cap=params['max_n'],
        embedding_field=params['embedding_field'],
    )
    await ctx.progress(0, len(ids), 'selecting')
    if not ids:
        return {
            'selection': {'crop_ids': [], 'method': 'kcenter_greedy', 'version': 'v1', 'n_pool': 0}
        }
    seed_idx: int | None = None
    if params['seed_crop_id'] is not None:
        with contextlib.suppress(ValueError):
            seed_idx = ids.index(params['seed_crop_id'])
    selected_idx: np.ndarray = k_center_greedy(
        l2_normalize(embeddings), params['k'], seed_idx=seed_idx
    )
    return {
        'selection': {
            'crop_ids': [ids[i] for i in selected_idx.tolist()],
            'method': 'kcenter_greedy',
            'version': 'v1',
            'n_pool': len(ids),
        }
    }


__all__ = ['KIND', 'run_selection_job', 'start', 'state_dict']
