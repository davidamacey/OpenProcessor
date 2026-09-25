"""Cluster-id maintenance helpers for the curation pipeline.

The labeler treats ``cluster_id`` as the grouping key on the
``/clusters`` page. For a class-clustered item ensemble the cluster
IS the class — any item with a known model / VLM / human class_id
should sit in its class's bucket. Without periodic normalization,
classes fragment across multiple cluster_ids depending on which
pipeline stage last touched the item.

This module is the post-prototype-deletion home of
``force_cluster_id_equals_class_id``. The previous prototype-cluster
module (and the prototype concept generally) was deleted because it
mis-labeled a large fraction of rows in production. ``cluster_id``
upkeep is a separate, narrowly-scoped concern that survives the
cleanup.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)

# Default poll cadence + ceiling for run_update_by_query_polled. 2s is
# frequent enough to feel responsive on a job dashboard without hammering
# the cluster; 1800s (30 min) matches the timeout envelope this module's
# blocking predecessor used for the same full-index operation.
_UBQ_POLL_INTERVAL_S = 2.0
_UBQ_POLL_TIMEOUT_S = 1800.0


async def run_update_by_query_polled(
    client: AsyncOpenSearch,
    *,
    index: str,
    body: dict[str, Any],
    conflicts: str = 'proceed',
    refresh: bool = True,
    poll_interval_s: float = _UBQ_POLL_INTERVAL_S,
    timeout_s: float = _UBQ_POLL_TIMEOUT_S,
) -> dict[str, Any]:
    """Run ``update_by_query`` without blocking on one giant response.

    ``wait_for_completion=True`` holds the HTTP connection open until
    OpenSearch finishes the *entire* operation, then returns one huge
    response. On a large index (the items index can be hundreds of
    thousands of docs) this has a real, observed failure mode distinct
    from a timeout: OpenSearch completes the operation successfully
    server-side (confirmed via ``GET _tasks`` mid-run), but the client
    fails to *parse* the final response — ``aiohttp.http_exceptions.
    BadHttpMessage: 400, Too many headers received`` — and
    opensearch-py's transport-level retry then re-runs the entire
    multi-minute operation again from scratch, hitting the identical
    parse failure every time. The operation itself was never broken;
    only the client's blocking-response path was.

    Fix: submit with ``wait_for_completion=False`` (returns a task id
    immediately, no giant response to fail on) and poll
    ``GET _tasks/{task_id}`` — the same task-status endpoint used to watch
    an in-flight ``update_by_query`` from the CLI — until OpenSearch
    reports it done. Raises on the task's own error (a real per-shard
    failure) or on exceeding ``timeout_s`` (mirrors the old client-side
    timeout, not a server-side one now).
    """
    started = await client.update_by_query(
        index=index,
        body=body,
        conflicts=conflicts,
        refresh=refresh,
        wait_for_completion=False,
    )
    task_id = started['task']
    deadline = time.monotonic() + timeout_s
    while True:
        task = await client.tasks.get(task_id=task_id)
        if task.get('completed'):
            error = task.get('error')
            if error:
                raise RuntimeError(f'update_by_query task {task_id} failed: {error}')
            return task.get('response') or {}
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f'update_by_query task {task_id} did not complete within {timeout_s}s'
            )
        await asyncio.sleep(poll_interval_s)


_SCOPE_CHUNK = 10_000
"""Crop ids per scoped update_by_query (well under the 65,536 terms cap)."""


def class_cluster_placement(update: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Cluster fields a class write should carry so the item lands in its
    class cluster at write time — the rule :func:`force_cluster_id_equals_class_id`
    applies in bulk, for one write (DQ-m3).

    Returns ``{'cluster_id': class_id, 'cluster_subid': None}`` when
    ``update`` sets a class from a confident source and ``current`` (the
    freshly read doc) is elsewhere and not excluded; ``{}`` otherwise.
    """
    from src.services.curation.ingest_class_sources import confident_class_sources

    class_id = update.get('class_id')
    if not isinstance(class_id, int) or isinstance(class_id, bool):
        return {}
    if update.get('class_source') not in confident_class_sources():
        return {}
    if current.get('class_excluded') or current.get('cluster_id') == class_id:
        return {}
    return {'cluster_id': class_id, 'cluster_subid': None}


async def force_cluster_id_equals_class_id(
    client: AsyncOpenSearch,
    *,
    crop_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Set ``cluster_id = class_id`` for every item where they disagree.

    ``crop_ids`` limits the pass to those items (a scoped job's
    selection); ``None`` is the whole index, ``[]`` touches nothing.
    """
    if crop_ids is None:
        return await _normalize(client, None)
    totals = {'status': 'ok', 'updated': 0, 'batches': 0, 'version_conflicts': 0, 'failures': 0}
    for i in range(0, len(crop_ids), _SCOPE_CHUNK):
        part = await _normalize(client, crop_ids[i : i + _SCOPE_CHUNK])
        if part.get('status') != 'ok':
            return part
        for key in ('updated', 'batches', 'version_conflicts', 'failures'):
            totals[key] += part[key]
    return totals


async def _normalize(client: AsyncOpenSearch, crop_ids: list[str] | None) -> dict[str, Any]:
    """One ``update_by_query`` pass over the index, or over ``crop_ids``.

    Implementation uses ``update_by_query`` with
    ``ctx._source`` semantics. This:

    * Operates on the document body (Python-dict-style ``.get`` access),
      so missing fields are simply ``null`` instead of an exception.
    * Skips no-change docs via ``ctx.op = 'noop'`` — no wasted I/O.
    * Returns ``{updated, batches, version_conflicts}`` for real progress
      reporting without client-side bookkeeping.
    * Matches the OS-recommended pattern for conditional bulk updates.

    Idempotent and safe to re-run.
    """
    from src.services.curation.clustering.embedding_reduce import CONFIDENT_CLASS_SOURCES

    config = get_curation_config()
    # `class_id` / `cluster_id` / `cluster_subid` here are the top-level
    # item fields (already generic — not the RegionFields-governed
    # region sub-annotation; see RegionFields' docstring scope).
    #
    # The old query (`exists: class_id` only) matched nearly every
    # doc — 234,575 on the legacy index — forcing OpenSearch to load and
    # reconstruct the full _source (including derived kNN vectors) only
    # for the script to noop. Worse, it isn't just wasteful: residual
    # crops sitting in a candidate cluster (cluster_id >= the residual
    # offset) carry an unconfident class_id, and this query pulled them
    # back out of their candidate cluster on every run, undoing the
    # residual pool's clustering. Narrow to confident/validated docs
    # (the same CONFIDENT_CLASS_SOURCES set the residual-pool builder in
    # embedding_reduce.py excludes — imported from there so the two sets
    # never drift apart) plus a doc-values script filter that excludes
    # already-correct docs at query time (no _source load). The
    # doc-values filter alone cut the legacy match from 234,575 to
    # 30,688 (7.6x); the class-source restriction narrows it further.
    body = {
        # Exclude class_excluded items. Exclusion keeps class_id but
        # sets cluster_id=-2 precisely so an excluded item drops out of its
        # class cluster; without this the normalizer pulls it straight back.
        'query': {
            'bool': {
                'filter': [
                    {'exists': {'field': 'class_id'}},
                    *([{'terms': {'crop_id': crop_ids}}] if crop_ids is not None else []),
                    {
                        'bool': {
                            'should': [
                                {'term': {'class_validated': True}},
                                {'terms': {'class_source': sorted(CONFIDENT_CLASS_SOURCES)}},
                            ],
                            'minimum_should_match': 1,
                        }
                    },
                    {
                        'script': {
                            'script': {
                                'source': (
                                    "doc['cluster_id'].size()==0 ||"
                                    " doc['cluster_id'].value != doc['class_id'].value"
                                )
                            }
                        }
                    },
                ],
                'must_not': [{'term': {'class_excluded': True}}],
            }
        },
        'script': {
            'source': (
                # `def` rather than `int` so we can hold either an int
                # or null without painless complaining. `Objects.equals`
                # handles the null case correctly.
                # Defense in depth: re-check class_excluded inside
                # the script too, in case a concurrent exclusion write
                # lands between the query match and this doc's update.
                'if (ctx._source.class_excluded == true) {'
                "  ctx.op = 'noop'; return;"
                '}'
                'def cid = ctx._source.class_id;'
                "if (cid == null) { ctx.op = 'noop'; return; }"
                'if (java.util.Objects.equals(ctx._source.cluster_id, cid)) {'
                "  ctx.op = 'noop';"
                '} else {'
                '  ctx._source.cluster_id = cid;'
                # cluster_subid is cluster-local; cluster_id changed,
                # so the prior sub-cluster grouping no longer applies.
                '  ctx._source.remove("cluster_subid");'
                '}'
            ),
            'lang': 'painless',
        },
    }
    try:
        # ``conflicts='proceed'`` because concurrent worker stages may
        # write to the same docs; version conflicts are recoverable on
        # the next pass.
        #
        # Polled, not blocking (run_update_by_query_polled) — on the
        # full items index (hundreds of thousands of docs) a blocking
        # ``wait_for_completion=True`` response is large enough to hit
        # a real, observed client-side parse failure ("Too many headers
        # received") even though the operation completes successfully
        # server-side every time; opensearch-py's transport then retries
        # the *entire* multi-minute operation from scratch, repeatedly,
        # hitting the identical failure. See the helper's docstring.
        resp = await run_update_by_query_polled(
            client,
            index=config.items_index,
            body=body,
            refresh=True,
            conflicts='proceed',
        )
    except Exception as exc:
        logger.warning('curation_force_cluster_eq_class_failed', error=str(exc))
        return {'status': 'error', 'error': str(exc)}

    return {
        'status': 'ok',
        'updated': int(resp.get('updated', 0)),
        'batches': int(resp.get('batches', 0)),
        'version_conflicts': int(resp.get('version_conflicts', 0)),
        'failures': len(resp.get('failures') or []),
    }
