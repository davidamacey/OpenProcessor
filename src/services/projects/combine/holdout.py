"""The target's frozen test split after a combine (owner decision D6).

``preserve_union`` keeps every image that was test in any source as test
(flagged while copying); this module records it. ``recompute`` picks a new
deterministic per-class split over the target's validated items instead."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.services.curation.holdout import (
    compute_holdout_sha,
    persist_freeze_record,
    select_test_holdout,
)


if TYPE_CHECKING:
    from pathlib import Path

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


def _freeze(target: ProjectRecord, crop_ids: list[str], spec: dict[str, Any]) -> Path | None:
    if not crop_ids:
        return None
    return persist_freeze_record(
        crop_ids=crop_ids,
        sha=compute_holdout_sha(crop_ids),
        cohort_spec=spec,
        per_class_counts={},
        state_dir=target.resources.project_state_dir / 'test_holdout',
        kind='import',
    )


async def record_union(
    client: Any, target: ProjectRecord, items_index: str, job_id: str
) -> dict[str, Any]:
    """Write the freeze record of every item flagged test while copying."""
    docs = await _scan(client, target, items_index, {'term': {'test_holdout': True}})
    ids = [str(d['crop_id']) for d in docs]
    path = _freeze(target, ids, {'kind': 'combine', 'job_id': job_id, 'holdout': 'preserve_union'})
    return {'holdout_items': len(ids), 'freeze_record': str(path) if path else None}


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
    path = _freeze(target, chosen, {'kind': 'combine', 'job_id': job_id, 'holdout': 'recompute'})
    return {
        'holdout_items': len(chosen),
        'per_class_counts': {str(k): v for k, v in per_class.items()},
        'freeze_record': str(path) if path else None,
    }


__all__ = ['recompute', 'record_union']
