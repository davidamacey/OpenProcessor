"""Curation router sub-module — ``POST /classes/merge``: fold one class into another."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Query
from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch.registry import ClassRegistryError
from src.routers.curation._class_models import ClassMergeRequest  # noqa: TC001 - FastAPI
from src.routers.curation._common import (
    OpenSearchDep,
    get_class_registry,
    items_index,
    labels_confirmed_index,
    logger,
    router,
)
from src.utils.timeutil import now_iso


async def _merge_dry_run(payload: ClassMergeRequest, opensearch: Any) -> dict[str, Any]:
    reg = get_class_registry()
    for cid in (payload.source_id, payload.target_id):
        if reg.get(cid) is None:
            raise HTTPException(status_code=400, detail=f'class_id {cid} not found')
    if payload.source_id == payload.target_id:
        raise HTTPException(status_code=400, detail='cannot merge a class into itself')
    of_source = {'term': {'class_id': payload.source_id}}
    not_holdout = {'term': {'test_holdout': True}}

    async def _count(query: dict[str, Any]) -> int:
        resp = await opensearch.count(index=items_index(), body={'query': query})
        return int(resp.get('count', 0))

    holdout = await _count({'bool': {'filter': [of_source, not_holdout]}})
    relabel = await _count({'bool': {'filter': [of_source], 'must_not': [not_holdout]}})
    # a merge no longer clears class_validated — a
    # human-validated crop of the source class stays validated under the
    # target. This count is who that carry-over applies to, not
    # (as the old 'would_unvalidate' name implied) who loses validation.
    validated = await _count(
        {
            'bool': {
                'filter': [of_source, {'term': {'class_validated': True}}],
                'must_not': [not_holdout],
            }
        }
    )
    return {
        'dry_run': True,
        'source_id': payload.source_id,
        'target_id': payload.target_id,
        'would_relabel': relabel,
        'validations_carried_over': validated,
        'holdout_blocking': holdout,
        'blocked': holdout > 0,
    }


@router.post('/classes/merge')
async def merge_class(
    payload: ClassMergeRequest,
    opensearch: OpenSearchDep,
    dry_run: Annotated[bool, Query(description='Report counts; change nothing.')] = False,
) -> dict[str, Any]:
    """Mark source deprecated; bulk-relabel matching crops + labels.

    ``dry_run=true`` returns ``{dry_run, source_id, target_id,
    would_relabel, validations_carried_over, holdout_blocking, blocked}``
    and writes nothing (a real merge 409s when ``holdout_blocking > 0``).
    A human-validated crop of the source class stays validated under the
    target — ``validations_carried_over`` counts how many crops keep their
    validation this way, not (unlike the retired ``would_unvalidate``
    field) how many lose it.
    """
    if dry_run:
        return await _merge_dry_run(payload, opensearch)
    # A merge that touches frozen test_holdout crops would relabel their
    # class_id and leave their recorded holdout identity (SHA1-of-crop_id-
    # per-class_id) stale. Refuse with 409 naming the affected count rather
    # than silently remapping. Checked before any registry mutation below.
    holdout_count_resp = await opensearch.count(
        index=items_index(),
        body={
            'query': {
                'bool': {
                    'filter': [
                        {'term': {'class_id': payload.source_id}},
                        {'term': {'test_holdout': True}},
                    ],
                },
            },
        },
    )
    holdout_count = int(holdout_count_resp.get('count', 0))
    if holdout_count:
        raise HTTPException(
            status_code=409,
            detail=(
                f'cannot merge class {payload.source_id}: {holdout_count} '
                'frozen test_holdout crop(s) would be relabeled'
            ),
        )
    reg = get_class_registry()
    try:
        result = reg.merge_class(payload.source_id, payload.target_id)
    except ClassRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target = reg.get(payload.target_id)
    target_name = target.class_name if target is not None else ''
    now = now_iso()
    relabel_doc = {
        'class_id': payload.target_id,
        'class_name': target_name,
        'class_source': 'class_merge',
        'updated_at': now,
    }
    merge_query = {
        'bool': {
            'filter': [{'term': {'class_id': payload.source_id}}],
            'must_not': [{'term': {'test_holdout': True}}],
        },
    }
    # The confirmed-labels index has no class_id_history / class_validated
    # field (its label_source means "original label provenance", not
    # "is this human-confirmed") — stays a plain update_by_query.
    #
    # This index has no mapping for cluster_id/cluster_subid — they
    # were only ever written here (nothing reads them off this index;
    # models.py's health check only checks index existence), so an
    # unmapped field was accumulating on every merge for no reader. Drop
    # both writes rather than add a mapping for dead fields.
    try:
        await opensearch.update_by_query(
            index=labels_confirmed_index(),
            body={
                'script': {
                    'source': (
                        'ctx._source.class_id = params.class_id;'
                        'ctx._source.class_name = params.class_name;'
                        'ctx._source.class_source = params.class_source;'
                        'ctx._source.updated_at = params.updated_at;'
                    ),
                    'params': relabel_doc,
                },
                'query': merge_query,
            },
            conflicts='proceed',
            refresh=True,
        )
    except Exception as exc:
        logger.warning('merge_relabel_failed', index=labels_confirmed_index(), error=str(exc))

    # Items index: a per-doc OCC bulk pass (rather than a bare
    # update_by_query painless script) so class_id_history gets appended
    # (dedupe/cap both live in record_class_history, reused here rather
    # than reimplemented in painless). A merged crop must not keep a stale
    # class_validated=true/label_source='human' once class_source now says
    # 'class_merge'.
    from src.clients.occ import occ_skip_on_conflict_bulk
    from src.domain.history import record_class_history

    async def _scroll_merge_ids() -> list[str]:
        ids: list[str] = []
        body = {'size': 500, 'query': merge_query, '_source': False}
        resp = await opensearch.search(index=items_index(), body=body, scroll='2m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']
        while hits:
            ids.extend(h['_id'] for h in hits)
            resp = await opensearch.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = resp.get('_scroll_id')
            hits = resp['hits']['hits']
        if scroll_id:
            try:
                await opensearch.clear_scroll(scroll_id=scroll_id)
            except OpenSearchException as exc:  # nosec B110 — advisory cleanup only
                logger.debug('merge_clear_scroll_failed', error=str(exc))
        return ids

    def _merge_crop(_doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        update = dict(relabel_doc)
        update['cluster_id'] = payload.target_id
        update['cluster_subid'] = None
        update['label_source'] = 'class_merge'
        # a human-validated crop of the source class stays
        # validated under the target — only the class assignment changes,
        # not the fact a human already confirmed it. class_id_history
        # (record_class_history below) separately snapshots the pre-merge
        # class_validated state, so the prior class's validation is on
        # record either way.
        update['class_validated'] = bool(current.get('class_validated', False))
        update['class_id_history'] = record_class_history(current, writer='class_merge')
        return update

    try:
        crop_ids = await _scroll_merge_ids()
        if crop_ids:
            # Refresh once after every page instead of forcing a
            # refresh=True on each of occ_skip_on_conflict_bulk's
            # per-500-id pages.
            bulk_result = await occ_skip_on_conflict_bulk(
                opensearch,
                doc_ids=crop_ids,
                merger=_merge_crop,
                index=items_index(),
                refresh=False,
                writer_id='class_merge',
            )
            if bulk_result.get('errors'):
                logger.warning(
                    'merge_relabel_partial_errors',
                    index=items_index(),
                    errors=len(bulk_result['errors']),
                )
            try:
                await opensearch.indices.refresh(index=items_index())
            except OpenSearchException as exc:
                logger.debug('merge_relabel_final_refresh_failed', error=str(exc))
    except Exception as exc:
        logger.warning('merge_relabel_failed', index=items_index(), error=str(exc))
    return result
