"""The queued ``item_scores`` job behind ``POST /curation/scores/compute``.

The route queues the job (one at a time per project); a Celery worker runs
:func:`run_scoring_job` over the requested scorers and returns the per-scorer results as the
job's result.

**Per-pool single-fetch design (load-bearing):** not every scorer wants the
same items. Each scorer declares a :attr:`~item_scores.base.CropScorer.pool`
(``'residual'`` or ``'probe_scored'`` — see ``base.py`` for the semantics),
and this module builds each *needed* pool exactly ONCE per job — never
unconditionally, and never once per scorer — then hands each scorer the
pool it asked for:

* ``'residual'`` — :func:`clustering.residual_fetch.fetch_residual_embeddings_parallel`,
  unchanged from the original single-pool design. The OpenSearch read (with
  embeddings) is the dominant cost here (350k crops x 1024-d f32 = 1.43 GB),
  not the math, so scorers sharing this pool (``uniqueness``, ``near_dup``)
  share one fetch. ``test_holdout=true`` crops are excluded via an explicit
  ``must_not`` (belt and suspenders — they're also implicitly excluded
  because ``fetch_residual_embeddings_parallel`` already drops
  ``class_validated: true`` crops, and a crop can only be
  ``test_holdout=true`` if it was ``class_validated=true`` at freeze time;
  see the reference review module's ``freeze_test_holdout``).
* ``'probe_scored'`` — ids only (no embeddings — a ``(n, 0)`` array is
  handed to the scorer), every item with ``probe_pred_class`` set, minus
  ``test_holdout`` / ``class_excluded`` / ``class_validated``. This is the
  pool ``mistakenness`` needs: it audits machine-confident labels via
  ``probe_pred_*`` scalar fields, never embeddings, so the residual pool
  (which excludes exactly the confidently-labeled items mistakenness
  exists to check) was the wrong pool for it (fixed 2026-09-25 — see
  ``git log`` for the incident: 1,351 residual items yielded
  ``n_scored=1``, ``n_skipped_outside_probe_classes=1350``, while 7,936
  items were actually probe-scored).

``progress.total`` / ``progress.done`` are the SUM of each requested
scorer's own pool size (a scorer sharing a pool with another requested
scorer counts that pool's size again — this is "work units across
requested scorers", not "distinct items touched this job"). Per-scorer
detail (which pool, how many items in it, how many it actually wrote) is
in ``result.results[scorer_name]`` — additive fields on the existing shape,
not a breaking change to ``/curation/scores/status``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.logging import get_logger
from src.services.curation.latest_job import common_state
from src.services.jobs.service import start_job


if TYPE_CHECKING:
    import uuid

    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repos.jobs import JobRow
    from src.schemas.jobs import JobView
    from src.services.curation.item_scores.base import ScoreResult
    from src.tasks.runtime.context import JobContext


logger = get_logger(__name__)

KIND = 'item_scores'


async def start(session: AsyncSession, project: uuid.UUID, scorer_names: list[str]) -> JobRow:
    """Queue the scoring job (409 ``job_busy`` while one is active)."""
    started = await start_job(
        session, KIND, project, params={'scorers': list(scorer_names)}, requested_by='api'
    )
    return started.job


def state_dict(view: JobView | None) -> dict[str, Any]:
    """The ``/scores/status`` wire of a job row."""
    progress = (view.progress if view else None) or {}
    return {
        **common_state(view),
        'scorers': list((view.params or {}).get('scorers', [])) if view else [],
        'processed': int(progress.get('done') or 0),
        'total': int(progress.get('total') or 0),
        'results': dict(((view.result or {}).get('results') or {}) if view else {}),
    }


async def bulk_write_result(
    opensearch: AsyncOpenSearch,
    result: ScoreResult,
    *,
    chunk_size: int = 500,
) -> int:
    """Write ``result`` back by bulk update; returns the number of items OpenSearch
    refused (a bulk answers 200 with per-item errors, which a discarded response hides)."""
    from src.clients.opensearch.bulk import failed_bulk_items
    from src.config.curation_accessors import get_curation_config
    from src.config.curation_roles import IndexRole, index_name

    index = index_name(get_curation_config(), IndexRole.ITEMS)
    items = list(result.fields.items())
    failed = 0
    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        bulk: list[dict[str, Any]] = []
        for crop_id, doc in chunk:
            bulk.append({'update': {'_index': index, '_id': crop_id}})
            bulk.append({'doc': doc})
        if bulk:
            resp = await opensearch.bulk(body=bulk, refresh=False)
            if isinstance(resp, dict):
                failed += len(failed_bulk_items(resp))
    if failed:
        logger.warning('curation_scores_bulk_item_errors', failed=failed, total=len(items))
    return failed


_PROBE_SCORED_PAGE_SIZE = 2000
_PROBE_SCORED_SCROLL_TTL = '2m'


async def _fetch_probe_scored_ids(client: AsyncOpenSearch) -> list[str]:
    """The ``'probe_scored'`` pool: ids only, no embeddings.

    Every item the probe has an opinion on (``exists probe_pred_class``),
    minus ``test_holdout`` (frozen eval set), ``class_excluded`` (human
    said "not usable"), and ``class_validated`` (human-validated labels are
    human-owned — not audited by an automated mistakenness pass). Full
    scroll, no cap — a silent partial pool here would silently under-audit
    machine labels, the exact failure mode this pool exists to fix.
    """
    from src.config.curation_accessors import get_curation_config
    from src.config.curation_roles import IndexRole, index_name

    index = index_name(get_curation_config(), IndexRole.ITEMS)
    query = {
        'bool': {
            'filter': [{'exists': {'field': 'probe_pred_class'}}],
            'must_not': [
                {'term': {'test_holdout': True}},
                {'term': {'class_excluded': True}},
                {'term': {'class_validated': True}},
            ],
        },
    }
    ids: list[str] = []
    body: dict[str, Any] = {
        'size': _PROBE_SCORED_PAGE_SIZE,
        'query': query,
        '_source': False,
    }
    resp = await client.search(index=index, body=body, scroll=_PROBE_SCORED_SCROLL_TTL)
    scroll_id = resp.get('_scroll_id')
    try:
        while True:
            hits = resp['hits']['hits']
            if not hits:
                break
            ids.extend(h['_id'] for h in hits)
            if not scroll_id:
                break
            resp = await client.scroll(scroll_id=scroll_id, scroll=_PROBE_SCORED_SCROLL_TTL)
            scroll_id = resp.get('_scroll_id')
    finally:
        if scroll_id:
            with contextlib.suppress(Exception):
                await client.clear_scroll(scroll_id=scroll_id)
    return ids


async def _build_pools(
    opensearch: AsyncOpenSearch,
    needed: set[str],
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Fetch each needed pool exactly once. ``needed`` is the set of
    ``pool`` values actually declared by the requested scorers — a pool
    nothing asked for is never fetched."""
    from src.services.curation.clustering.residual_fetch import fetch_residual_embeddings_parallel

    pools: dict[str, tuple[list[str], np.ndarray]] = {}
    if 'residual' in needed:
        pools['residual'] = await fetch_residual_embeddings_parallel(
            opensearch,
            # Belt-and-suspenders test_holdout exclusion — see module
            # docstring for why this is already implied.
            extra_must=[{'bool': {'must_not': [{'term': {'test_holdout': True}}]}}],
        )
    if 'probe_scored' in needed:
        probe_ids = await _fetch_probe_scored_ids(opensearch)
        pools['probe_scored'] = (probe_ids, np.zeros((len(probe_ids), 0), dtype=np.float32))
    return pools


async def run_scoring_job(
    ctx: JobContext, opensearch: AsyncOpenSearch, scorer_names: list[str]
) -> dict[str, Any]:
    """Actual compute body: fetch each pool the requested scorers need (once each), then one
    scorer at a time, writing results back by bulk update and reporting progress between
    scorers. A cancel is seen between scorers."""
    from src.services.curation.item_scores import get_scorer

    scorers = {name: get_scorer(name) for name in scorer_names}
    pools = await _build_pools(opensearch, {scorer.pool for scorer in scorers.values()})
    total = sum(len(pools[scorer.pool][0]) for scorer in scorers.values())
    await ctx.progress(0, total, 'scoring')
    results: dict[str, Any] = {}
    processed = 0
    for name in scorer_names:
        if ctx.cancelled():
            break
        scorer = scorers[name]
        ids, embeddings = pools[scorer.pool]
        result = await scorer.score(ids, embeddings, opensearch=opensearch)
        n_failed = await bulk_write_result(opensearch, result)
        if n_failed and n_failed >= len(result.fields):
            msg = f'{name}: no score was written ({n_failed} bulk item errors)'
            raise RuntimeError(msg)
        results[name] = {
            'n_scored': result.n_scored,
            'n_failed': n_failed,
            'extra': result.extra,
            'pool': scorer.pool,
            'pool_size': len(ids),
        }
        processed += len(ids)
        await ctx.progress(processed, total, name)
    return {'results': results, 'cancelled': ctx.cancelled()}


async def compute_coverage(opensearch: AsyncOpenSearch) -> dict[str, Any]:
    """Per-scorer ``{n_scored, total, pct}`` for ``GET /curation/scores/coverage``."""
    from src.config.curation_accessors import get_curation_config
    from src.config.curation_roles import IndexRole, index_name
    from src.services.curation.item_scores import SCORER_METADATA

    index = index_name(get_curation_config(), IndexRole.ITEMS)
    total_resp = await opensearch.count(index=index, body={'query': {'match_all': {}}})
    total = int(total_resp.get('count', 0))

    coverage: dict[str, Any] = {}
    for scorer_id, meta in SCORER_METADATA.items():
        field_name = meta['requires_field']
        try:
            resp = await opensearch.count(
                index=index, body={'query': {'exists': {'field': field_name}}}
            )
            n = int(resp.get('count', 0))
        except Exception as exc:
            logger.warning(
                'curation_scores_coverage_count_failed', scorer=scorer_id, error=str(exc)
            )
            n = 0
        coverage[scorer_id] = {
            'field': field_name,
            'n_scored': n,
            'total': total,
            'pct': round(100.0 * n / total, 2) if total else 0.0,
        }
    return coverage


__all__ = [
    'KIND',
    'bulk_write_result',
    'compute_coverage',
    'run_scoring_job',
    'start',
    'state_dict',
]
