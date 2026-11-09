"""Project registry changes onto the items index: a merge's relabel and the name outbox.

The registry is PostgreSQL; the items carry a denormalised ``class_id`` / ``class_name`` pair.
A rename or a merge writes a ``search_projection_tasks`` row in the registry's own transaction
(D-47); the ``class_projection_sync`` job applies that row with one ``update_by_query`` filtered
by ``class_id`` and then sets ``done_at``. Every query below only selects items whose pair still
disagrees with the registry, so a re-run changes nothing. The reconciliation the upgrade's
verifier reports (``class_name`` disagreeing with the registry) is the same predicate.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.clients.occ import occ_skip_on_conflict_bulk
from src.clients.opensearch.scroll import scroll_ids
from src.clients.opensearch.tasks import run_update_by_query_polled
from src.domain.history import record_class_history
from src.services.embedding_spaces.readers import vector_source_excludes
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

HOLDOUT = {'term': {'test_holdout': True}}
MERGE_WRITER = 'class_merge'


class ProjectionTaskError(RuntimeError):
    """A projection task whose payload this job cannot apply (it stays pending, fail closed)."""


def stale_name_query(class_id: int, class_name: str) -> dict[str, Any]:
    """Items of ``class_id`` whose stored ``class_name`` is not ``class_name``."""
    return {
        'bool': {
            'filter': [{'term': {'class_id': class_id}}],
            'must_not': [{'term': {'class_name': class_name}}],
        }
    }


def merge_source_query(source_ids: list[int]) -> dict[str, Any]:
    """Items still carrying a merged-away class id (the frozen holdout is never relabelled)."""
    return {
        'bool': {'filter': [{'terms': {'class_id': source_ids}}], 'must_not': [HOLDOUT]},
    }


def holdout_of_class_query(class_id: int) -> dict[str, Any]:
    return {'bool': {'filter': [{'term': {'class_id': class_id}}, HOLDOUT]}}


async def count_items(opensearch: Any, index: str, query: dict[str, Any]) -> int:
    response = await opensearch.count(index=index, body={'query': query})
    return int(response.get('count', 0))


_RENAME_SCRIPT = {
    'lang': 'painless',
    'source': 'ctx._source.class_name = params.name; ctx._source.updated_at = params.now',
}
_MERGE_SCRIPT = {
    'lang': 'painless',
    'source': (
        'ctx._source.class_id = params.target_id; ctx._source.class_name = params.name; '
        "ctx._source.class_source = 'class_merge'; ctx._source.updated_at = params.now"
    ),
}


async def apply_rename(opensearch: Any, index: str, class_id: int, new_name: str) -> int:
    """Set ``class_name`` on every item of ``class_id`` that still has another name."""
    body = {
        'query': stale_name_query(class_id, new_name),
        'script': {**_RENAME_SCRIPT, 'params': {'name': new_name, 'now': now_iso()}},
    }
    done = await run_update_by_query_polled(opensearch, index=index, body=body)
    return int(done.get('updated', 0))


async def apply_merge(
    opensearch: Any, index: str, source_ids: list[int], target_id: int, target_name: str
) -> int:
    """Move the items still on a merged-away id to the target (a catch-up after the job)."""
    body = {
        'query': merge_source_query(source_ids),
        'script': {
            **_MERGE_SCRIPT,
            'params': {'target_id': target_id, 'name': target_name, 'now': now_iso()},
        },
    }
    done = await run_update_by_query_polled(opensearch, index=index, body=body)
    return int(done.get('updated', 0))


async def apply_task(
    opensearch: Any,
    index: str,
    kind: str,
    payload: dict[str, Any],
    name_of: Callable[[int], str | None],
) -> int:
    """Apply one outbox row; ``name_of`` reads the registry's current name of a class id."""
    if kind == 'class_rename':
        # The registry's current name wins over the payload's: two renames in a row converge.
        class_id = int(payload['class_id'])
        current = name_of(class_id) or str(payload['new_name'])
        return await apply_rename(opensearch, index, class_id, current)
    if kind == 'class_merge':
        target_id = int(payload['target_id'])
        name = name_of(target_id)
        if name is None:
            raise ProjectionTaskError(f'merge target class_id {target_id} is not in the registry')
        return await apply_merge(
            opensearch, index, [int(i) for i in payload['source_ids']], target_id, name
        )
    raise ProjectionTaskError(f'class_projection_sync cannot apply a {kind!r} row')


async def mark_done(session: AsyncSession, task_id: uuid.UUID) -> None:
    await session.execute(
        text(
            'UPDATE search_projection_tasks SET done_at = now() WHERE id = :id AND done_at IS NULL'
        ),
        {'id': task_id},
    )


async def task_row(session: AsyncSession, task_id: uuid.UUID) -> tuple[str, dict[str, Any], bool]:
    """``(kind, payload, done)`` of one outbox row; raises when it does not exist."""
    found = (
        await session.execute(
            text(
                'SELECT kind, payload, done_at IS NOT NULL FROM search_projection_tasks WHERE id = :id'
            ),
            {'id': task_id},
        )
    ).first()
    if found is None:
        raise ProjectionTaskError(f'no projection task {task_id}')
    return found[0], dict(found[1] or {}), bool(found[2])


def _merge_crop_merger(
    target_id: int, target_name: str
) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    relabel = {
        'class_id': target_id,
        'class_name': target_name,
        'class_source': 'class_merge',
        'label_source': 'class_merge',
    }

    def merge(_doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        # A human-validated crop of the source class stays validated under the target; the
        # history snapshot records the pre-merge state either way.
        return {
            **relabel,
            'updated_at': now_iso(),
            'class_validated': bool(current.get('class_validated', False)),
            'class_id_history': record_class_history(current, writer=MERGE_WRITER),
        }

    return merge


async def relabel_merged_items(
    opensearch: Any,
    index: str,
    source_id: int,
    target_id: int,
    target_name: str,
    *,
    cancelled: Callable[[], bool],
) -> tuple[int, int]:
    """``(items, errors)`` of the per-document OCC pass that appends ``class_id_history``.

    The pass is idempotent: it selects the items still on ``source_id``. A bulk error is
    returned to the caller, which fails the job instead of reporting a half merge as done.
    """
    ids = await scroll_ids(opensearch, index=index, query=merge_source_query([source_id]))
    if not ids or cancelled():
        return 0, 0
    result = await occ_skip_on_conflict_bulk(
        opensearch,
        vector_excludes=await vector_source_excludes(),
        doc_ids=ids,
        merger=_merge_crop_merger(target_id, target_name),
        index=index,
        refresh=True,
        writer_id=MERGE_WRITER,
    )
    return len(ids), len(result.get('errors') or [])


# --- the two job bodies ------------------------------------------------------------------------


async def run_class_merge(ctx: Any, opensearch: Any) -> dict[str, Any]:
    """The ``class_merge`` job: registry merge (once), then the history-keeping relabel.

    Both halves are idempotent, so an interrupted run that is started again finishes the merge.
    A frozen holdout crop of the source class refuses the whole merge before anything is written.
    """
    from src.config.curation_accessors import items_index
    from src.services.classes.registry import get_class_registry

    source_id = int(ctx.params['source_id'])
    target_id = int(ctx.params['target_id'])
    index = items_index()
    held = await count_items(opensearch, index, holdout_of_class_query(source_id))
    if held:
        raise ProjectionTaskError(
            f'cannot merge class {source_id}: {held} frozen test_holdout crop(s) would be relabeled'
        )
    registry = get_class_registry()
    doc = await registry.snapshot()
    source = doc.get(source_id)
    if source is None or doc.get(target_id) is None:
        raise ProjectionTaskError(f'class {source_id} or {target_id} is not in the registry')
    if source.merged_into != target_id:
        await registry.merge_class(source_id, target_id)
        doc = await registry.snapshot()
    target = doc.get(target_id)
    assert target is not None
    await ctx.progress(0, None, 'relabel')
    items, errors = await relabel_merged_items(
        opensearch, index, source_id, target_id, target.class_name, cancelled=ctx.cancelled
    )
    await ctx.progress(items, items, 'relabel')
    return {'source_id': source_id, 'target_id': target_id, 'relabelled': items, 'errors': errors}


async def run_projection_sync(ctx: Any, opensearch: Any) -> dict[str, Any]:
    """The ``class_projection_sync`` job: apply the one outbox row named in ``params``."""
    from src.config.curation_accessors import items_index
    from src.services.classes.registry import get_class_registry

    task_id = uuid.UUID(str(ctx.params['projection_task_id']))
    async with ctx.session() as session:
        kind, payload, done = await task_row(session, task_id)
    if done:
        return {'task': str(task_id), 'updated': 0, 'already_done': True}
    doc = await get_class_registry().snapshot()

    def name_of(class_id: int) -> str | None:
        entry = doc.get(class_id)
        return None if entry is None else entry.class_name

    updated = await apply_task(opensearch, items_index(), kind, payload, name_of)
    async with ctx.session() as session:
        await mark_done(session, task_id)
    return {'task': str(task_id), 'kind': kind, 'updated': updated}
