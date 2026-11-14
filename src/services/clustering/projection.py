"""``cluster_projection_sync``: write a partition's assignments onto the items (plan 5.3, 7.1).

PostgreSQL holds the partition; OpenSearch holds the served copy as search data. This is the one
writer of the cluster fields. Fences, so an older projection can never overwrite a newer one:

* before every batch the job re-reads ``pending_partition_id`` and stops when it is no longer its
  partition (nothing a newer activation could lose is written afterwards);
* every document update carries the partition id and applies only if the document's stored
  ``cluster_partition_id`` is not newer (partition ids are UUIDv7, so text order is time order).

The flip of ``serving_partition_id`` happens last, in ``activation.complete_projection``.
Item keys are crop ids; a region-box key is ``crop_id:box_id`` and is updated inside the item's
``region_boxes`` list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.core.logging import get_logger
from src.domain.nested_limit import is_over_limit_bulk_error, painless_guard
from src.services.clustering.activation import complete_projection


if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable, Callable

    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

BATCH = 1000
BOX_SEPARATOR = ':'
ASSIGNED_BY_RUN = 'run'

# Both scripts run only on an item under the nested-document limit (src/domain/nested_limit.py,
# OpenSearch k-NN issue 3623): a scripted update rewrites the whole document, and on an item at
# the limit the stored box vectors it would write back are another document's. An item over the
# limit fails its bulk item with a reason `_bulk` recognises and counts apart from real failures.
_ITEM_BODY = """
if (ctx._source.cluster_partition_id != null && ctx._source.cluster_partition_id.compareTo(params.pid) > 0) {
  ctx.op = 'noop';
} else {
  ctx._source.cluster_partition_id = params.pid;
  ctx._source.cluster_uid = params.uid;
  ctx._source.cluster_state = params.state;
  ctx._source.cluster_score = params.score;
  ctx._source.cluster_margin = params.margin;
  ctx._source.cluster_assigned_by = params.by;
}
"""
_ITEM_SCRIPT = painless_guard(_ITEM_BODY, on_over='throw')
_BOX_BODY = """
if (ctx._source.region_boxes == null) { ctx.op = 'noop'; } else {
  for (def box : ctx._source.region_boxes) {
    if (box.box_id == params.box_id) {
      if (box.cluster_partition_id != null && box.cluster_partition_id.compareTo(params.pid) > 0) { continue; }
      box.cluster_partition_id = params.pid;
      box.cluster_uid = params.uid;
      box.cluster_state = params.state;
      box.cluster_score = params.score;
      box.cluster_margin = params.margin;
      box.cluster_assigned_by = params.by;
    }
  }
}
"""
_BOX_SCRIPT = painless_guard(_BOX_BODY, on_over='throw')

_PAGE = text(
    'SELECT item_key, cluster_uid, state, score, margin FROM cluster_memberships '
    'WHERE partition_id = :id AND item_key > :after ORDER BY item_key LIMIT :n'
)
_GONE_PAGE = text(
    'SELECT o.item_key FROM cluster_memberships o WHERE o.partition_id = :old '
    'AND o.item_key > :after AND NOT EXISTS (SELECT 1 FROM cluster_memberships n '
    'WHERE n.partition_id = :id AND n.item_key = o.item_key) ORDER BY o.item_key LIMIT :n'
)


@dataclass(frozen=True)
class ProjectionResult:
    status: str
    """``flipped``, or ``superseded`` when a newer activation replaced this partition."""
    written: int = 0
    cleared: int = 0
    failed: int = 0
    skipped_over_nested_limit: int = 0
    """Items left as they were: they hold too many nested documents (the count of items, not
    of writes; a warning is logged once with it)."""


def update_action(
    index: str, target: str, key: str, partition_id: uuid.UUID, row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One ``_bulk`` update (action line, body line) with the fence script."""
    params = {'pid': str(partition_id), 'by': ASSIGNED_BY_RUN, **row}
    if target == 'region_box':
        crop_id, _, box_id = key.partition(BOX_SEPARATOR)
        return (
            {'update': {'_index': index, '_id': crop_id, 'retry_on_conflict': 3}},
            {
                'script': {
                    'lang': 'painless',
                    'source': _BOX_SCRIPT,
                    'params': {**params, 'box_id': box_id},
                }
            },
        )
    return (
        {'update': {'_index': index, '_id': key, 'retry_on_conflict': 3}},
        {'script': {'lang': 'painless', 'source': _ITEM_SCRIPT, 'params': params}},
    )


def _row(
    uid: uuid.UUID | None, state: str, score: float | None, margin: float | None
) -> dict[str, Any]:
    return {
        'uid': None if uid is None else str(uid),
        'state': state,
        'score': score,
        'margin': margin,
    }


async def _still_pending(session: AsyncSession, partition_id: uuid.UUID) -> bool:
    row = await session.execute(
        text('SELECT 1 FROM cluster_activations WHERE pending_partition_id = :id'),
        {'id': partition_id},
    )
    return row.first() is not None


async def _bulk(
    client: AsyncOpenSearch, actions: list[tuple[dict[str, Any], dict[str, Any]]]
) -> tuple[int, int]:
    """Send one batch; return ``(failed items, items skipped over the nested limit)``."""
    if not actions:
        return 0, 0
    body = [line for pair in actions for line in pair]
    response = await client.bulk(body=body, refresh=False)
    if not response.get('errors'):
        return 0, 0
    failed = skipped = 0
    for item in response['items']:
        error = item.get('update', {}).get('error')
        if error is None:
            continue
        if is_over_limit_bulk_error(error):
            skipped += 1
        else:
            failed += 1
    return failed, skipped


async def project_partition(
    session: AsyncSession,
    client: AsyncOpenSearch,
    *,
    project: uuid.UUID,
    partition_id: uuid.UUID,
    index: str,
    target: str,
    old_partition_id: uuid.UUID | None,
    on_batch: Callable[[int], Awaitable[None]] | None = None,
) -> ProjectionResult:
    """Project ``partition_id`` onto ``index`` and flip it to serving; fenced as described above."""
    written = cleared = failed = over_limit = 0
    after = ''
    while True:
        if not await _still_pending(session, partition_id):
            return ProjectionResult('superseded', written, cleared, failed, over_limit)
        page = (
            await session.execute(_PAGE, {'id': partition_id, 'after': after, 'n': BATCH})
        ).all()
        if not page:
            break
        batch_failed, batch_over = await _bulk(
            client,
            [
                update_action(index, target, key, partition_id, _row(uid, state, score, margin))
                for key, uid, state, score, margin in page
            ],
        )
        failed += batch_failed
        over_limit += batch_over
        written += len(page)
        after = page[-1][0]
        if on_batch is not None:
            await on_batch(written)
    if old_partition_id is not None:
        after = ''
        while True:
            if not await _still_pending(session, partition_id):
                return ProjectionResult('superseded', written, cleared, failed, over_limit)
            gone = (
                await session.execute(
                    _GONE_PAGE,
                    {'old': old_partition_id, 'id': partition_id, 'after': after, 'n': BATCH},
                )
            ).all()
            if not gone:
                break
            batch_failed, batch_over = await _bulk(
                client,
                [
                    update_action(
                        index, target, key, partition_id, _row(None, 'unassigned', None, None)
                    )
                    for (key,) in gone
                ],
            )
            failed += batch_failed
            over_limit += batch_over
            cleared += len(gone)
            after = gone[-1][0]
    if over_limit:
        logger.warning(
            'cluster_projection_skipped_over_nested_limit',
            partition_id=str(partition_id),
            items=over_limit,
        )
    if failed:
        return ProjectionResult('failed', written, cleared, failed, over_limit)
    flipped = await complete_projection(session, project=project, partition_id=partition_id)
    await session.execute(
        text(
            "UPDATE search_projection_tasks SET done_at = now() WHERE kind = 'cluster_partition' "
            "AND done_at IS NULL AND payload->>'partition_id' = :id"
        ),
        {'id': str(partition_id)},
    )
    return ProjectionResult(
        'flipped' if flipped else 'superseded', written, cleared, failed, over_limit
    )


__all__ = ['BATCH', 'ProjectionResult', 'project_partition', 'update_action']
