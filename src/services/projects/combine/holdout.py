"""The target's frozen test split after a combine (owner decision D6).

``preserve_union`` keeps every image that was test in any source as test
(flagged while copying); this module records it. ``recompute`` picks a new
deterministic per-class split over the target's validated items instead."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.db.session import async_session_factory
from src.services.curation.holdout import record_standing_freeze, select_test_holdout


if TYPE_CHECKING:
    import uuid

    from src.config.projects import ProjectRecord

_PAGE = 500


async def _scan(
    client: Any, target: ProjectRecord, index: str, query: dict[str, Any]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor: list[Any] | None = None
    with bind_project(target):
        while True:
            body: dict[str, Any] = {'size': _PAGE, 'query': query, 'sort': [{'crop_id': 'asc'}]}
            if cursor is not None:
                body['search_after'] = cursor
            hits = ((await client.search(index=index, body=body)).get('hits') or {}).get(
                'hits'
            ) or []
            out.extend(h.get('_source') or {} for h in hits)
            if len(hits) < _PAGE:
                return out
            cursor = hits[-1].get('sort')


async def _freeze(
    target: ProjectRecord, crop_ids: list[str], spec: dict[str, Any], job_id: str
) -> uuid.UUID | None:
    if not crop_ids:
        return None
    with bind_project(target):
        async with async_session_factory()() as session:
            row = await record_standing_freeze(
                session,
                crop_ids=crop_ids,
                cohort_spec=spec,
                per_class_counts={},
                source=f'combine:{job_id}',
                frozen_by='combine',
            )
    return row.id


async def record_union(
    client: Any, target: ProjectRecord, items_index: str, job_id: str
) -> dict[str, Any]:
    """Write the freeze record of every item flagged test while copying."""
    docs = await _scan(client, target, items_index, {'term': {'test_holdout': True}})
    ids = [str(d['crop_id']) for d in docs]
    freeze_id = await _freeze(
        target, ids, {'kind': 'combine', 'job_id': job_id, 'holdout': 'preserve_union'}, job_id
    )
    return {'holdout_items': len(ids), 'freeze_record': str(freeze_id) if freeze_id else None}


async def recompute(
    client: Any, target: ProjectRecord, items_index: str, job_id: str
) -> dict[str, Any]:
    """Choose a fresh test split: per class, a deterministic sample of the
    target's validated items. Items outside it are trained on."""
    docs = await _scan(client, target, items_index, {'term': {'class_validated': True}})
    by_class: dict[Any, list[str]] = {}
    for d in docs:
        by_class.setdefault(d.get('class_id'), []).append(str(d['crop_id']))
    chosen, per_class = select_test_holdout(by_class)
    picked = set(chosen)
    rewrite = [d for d in docs if bool(d.get('test_holdout')) != (str(d['crop_id']) in picked)]
    with bind_project(target):
        for start in range(0, len(rewrite), _PAGE):
            body: list[dict[str, Any]] = []
            for d in rewrite[start : start + _PAGE]:
                body.append({'index': {'_index': items_index, '_id': d['crop_id']}})
                body.append({**d, 'test_holdout': str(d['crop_id']) in picked})
            await client.bulk(body=body, refresh=False)
    freeze_id = await _freeze(
        target, chosen, {'kind': 'combine', 'job_id': job_id, 'holdout': 'recompute'}, job_id
    )
    return {
        'holdout_items': len(chosen),
        'per_class_counts': {str(k): v for k, v in per_class.items()},
        'freeze_record': str(freeze_id) if freeze_id else None,
    }


__all__ = ['recompute', 'record_union']
