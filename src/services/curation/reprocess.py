"""Unified reprocess (W10.13): one function, one lock rule.

Re-running something on an item or image used to be three tools (a region
requeue, a ``clear_detection`` wipe, a retry of ``detection_failed``). This
module is the single entry point; the scopes are:

=========  =======  ==========================================================
scope      unit     what it regenerates
=========  =======  ==========================================================
detect     image    machine items from the ingest detectors (merged under the
                    lock rule: ``reprocess_detect``)
region     item     machine region boxes (``reprocess_region``)
vlm        item     the VLM's class answer (``reprocess_vlm``)
embed      image    PE crop / frame / region vectors (``reprocess_embed``)
=========  =======  ==========================================================

Human and imported labels and boxes are locked and are never written; the
predicates live in :mod:`~src.services.curation.reprocess_locks`. Targets are
exactly one of explicit ``crop_ids``, explicit ``image_ids`` or a ``filter``
(:mod:`~src.services.curation.reprocess_targets`).

:func:`plan_reprocess` is read-only and returns the per-scope counts a dry
run serves. :func:`apply_reprocess` plans, then executes: ``region`` and
``vlm`` are OCC bulk flips done in the call; ``detect`` / ``embed`` run in
the call for a few images and as a file-backed job
(:mod:`~src.services.curation.reprocess_job`) above
``OP_REPROCESS_SYNC_MAX`` images.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.services.curation.dataset_import.limits import reprocess_sync_max
from src.services.curation.reprocess_detect import redetect_image
from src.services.curation.reprocess_embed import reembed_items
from src.services.curation.reprocess_images import IMAGE_SCOPES, process_images
from src.services.curation.reprocess_job import read_job, start_job
from src.services.curation.reprocess_locks import class_locked, class_locked_clause, item_locked
from src.services.curation.reprocess_models import (
    ReprocessRequest,
    ReprocessResponse,
    ReprocessScope,
    ReprocessScopeResult,
)
from src.services.curation.reprocess_region import (
    apply_region_filter,
    apply_region_ids,
    plan_region_filter,
    region_items,
    split_locked,
)
from src.services.curation.reprocess_targets import (
    ReprocessTargetsError,
    existing_images,
    item_filter_query,
    items_by_terms,
    scan_items,
    validate_targets,
)
from src.services.curation.reprocess_vlm import apply_vlm, vlm_class_clause, vlm_restore_update


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.ingest import CurationIngestService


# Execution order: detection first, derived vectors last, so an embed pass
# sees the items a detect pass just wrote.
_ORDER: tuple[ReprocessScope, ...] = ('detect', 'region', 'vlm', 'embed')

ServiceFactory = Callable[[], Awaitable['CurationIngestService']]


@dataclass
class ReprocessPlan:
    kind: str
    results: list[ReprocessScopeResult]
    image_ids: list[str] = field(default_factory=list)
    """Resolved images for the image-unit scopes (empty if none requested)."""
    not_found_crops: int = 0


def _ordered(scopes: list[ReprocessScope]) -> list[ReprocessScope]:
    return [s for s in _ORDER if s in scopes]


def _cfg_indexes() -> tuple[str, str]:
    cfg = get_curation_config()
    return cfg.items_index, cfg.images_index


async def _crop_docs(
    opensearch: AsyncOpenSearch, ids: list[str], includes: list[str]
) -> list[tuple[str, dict[str, Any]]]:
    return await items_by_terms(
        opensearch, 'crop_id', ids, index=_cfg_indexes()[0], includes=includes
    )


async def resolve_image_ids(
    opensearch: AsyncOpenSearch, request: ReprocessRequest, kind: str
) -> tuple[list[str], int]:
    """The images the image-unit scopes run on, deduplicated in a stable
    order, and how many requested crop ids named no item."""
    targets = request.targets
    if kind == 'image_ids':
        return list(dict.fromkeys(targets.image_ids or [])), 0
    if kind == 'crop_ids':
        wanted = list(dict.fromkeys(targets.crop_ids or []))
        docs = await _crop_docs(opensearch, wanted, ['image_id'])
        images = [src['image_id'] for _, src in docs if src.get('image_id')]
        return list(dict.fromkeys(images)), len(wanted) - len(docs)
    assert targets.filter is not None
    hits = await scan_items(
        opensearch,
        item_filter_query(targets.filter),
        index=_cfg_indexes()[0],
        includes=['image_id'],
    )
    return sorted({src['image_id'] for _, src in hits if src.get('image_id')}), 0


async def _plan_region(
    opensearch: AsyncOpenSearch, request: ReprocessRequest, kind: str
) -> tuple[ReprocessScopeResult, list[str]]:
    """``(result, unlocked item ids for explicit targets)``."""
    targets = request.targets
    if kind == 'filter':
        assert targets.filter is not None
        return await plan_region_filter(opensearch, targets.filter, request.region_mode), []
    field_name = 'crop_id' if kind == 'crop_ids' else 'image_id'
    wanted = list(dict.fromkeys(getattr(targets, kind)))
    docs = await region_items(opensearch, field_name, wanted)
    unlocked, locked = split_locked(docs)
    result = ReprocessScopeResult(
        scope='region',
        selected=len(docs),
        locked_skipped=len(locked),
        not_found=len(wanted) - len(docs) if kind == 'crop_ids' else 0,
    )
    return result, unlocked


async def _plan_vlm(
    opensearch: AsyncOpenSearch, request: ReprocessRequest, kind: str
) -> tuple[ReprocessScopeResult, list[str] | None]:
    """``(result, eligible ids)``; ``None`` ids means "page the filter"."""
    targets = request.targets
    items_index = _cfg_indexes()[0]
    if kind == 'filter':
        assert targets.filter is not None
        base = item_filter_query(targets.filter)

        async def count(*extra: dict[str, Any], not_locked: bool = False) -> int:
            query = {
                'bool': {
                    'filter': [base, *extra],
                    'must_not': [class_locked_clause()] if not_locked else [],
                }
            }
            return int((await opensearch.count(index=items_index, body={'query': query}))['count'])

        total = await count()
        locked = await count(class_locked_clause())
        eligible = await count(vlm_class_clause(), not_locked=True)
        return (
            ReprocessScopeResult(
                scope='vlm', selected=total, locked_skipped=locked, detail={'eligible': eligible}
            ),
            None,
        )
    field_name = 'crop_id' if kind == 'crop_ids' else 'image_id'
    wanted = list(dict.fromkeys(getattr(targets, kind)))
    docs = await items_by_terms(
        opensearch,
        field_name,
        wanted,
        index=items_index,
        includes=[
            'class_source',
            'class_validated',
            'test_holdout',
            'class_id_history',
        ],
    )
    eligible_ids = [cid for cid, src in docs if vlm_restore_update(src, now='') is not None]
    return (
        ReprocessScopeResult(
            scope='vlm',
            selected=len(docs),
            locked_skipped=sum(1 for _, src in docs if class_locked(src)),
            not_found=len(wanted) - len(docs) if kind == 'crop_ids' else 0,
            detail={'eligible': len(eligible_ids)},
        ),
        eligible_ids,
    )


async def _plan_images(
    opensearch: AsyncOpenSearch, scope: ReprocessScope, image_ids: list[str], kind: str, nf: int
) -> ReprocessScopeResult:
    items_index, images_index = _cfg_indexes()
    result = ReprocessScopeResult(scope=scope, selected=len(image_ids), not_found=nf)
    if kind == 'image_ids':
        found = await existing_images(opensearch, image_ids, index=images_index)
        result.not_found = len(image_ids) - len(found)
    if scope == 'detect' and image_ids:
        docs = await items_by_terms(
            opensearch,
            'image_id',
            image_ids,
            index=items_index,
            includes=_lock_includes(),
        )
        result.locked_skipped = sum(1 for _, src in docs if item_locked(src))
    return result


def _lock_includes() -> list[str]:
    from src.config.region_fields import get_region_fields

    F = get_region_fields()
    return ['class_source', 'class_validated', 'test_holdout', F.boxes, F.validated, F.verifier]


async def plan_reprocess(opensearch: AsyncOpenSearch, request: ReprocessRequest) -> ReprocessPlan:
    """Read-only: resolve the targets and count, per scope, what is
    selected and what the lock rule skips. Raises
    :class:`~...reprocess_targets.ReprocessTargetsError` for malformed
    targets."""
    kind = validate_targets(request.targets)
    scopes = _ordered(request.scopes)
    plan = ReprocessPlan(kind=kind, results=[])
    if any(s in IMAGE_SCOPES for s in scopes):
        plan.image_ids, plan.not_found_crops = await resolve_image_ids(opensearch, request, kind)
    for scope in scopes:
        if scope == 'region':
            plan.results.append((await _plan_region(opensearch, request, kind))[0])
        elif scope == 'vlm':
            plan.results.append((await _plan_vlm(opensearch, request, kind))[0])
        else:
            plan.results.append(
                await _plan_images(opensearch, scope, plan.image_ids, kind, plan.not_found_crops)
            )
    return plan


async def apply_reprocess(
    opensearch: AsyncOpenSearch,
    request: ReprocessRequest,
    *,
    service_factory: ServiceFactory | None = None,
) -> ReprocessResponse:
    """Plan, then (unless ``dry_run``) execute. ``service_factory`` builds
    the ingest service ``detect`` / ``embed`` need; it is only awaited when
    one of those scopes runs."""
    plan = await plan_reprocess(opensearch, request)
    if request.dry_run:
        return ReprocessResponse(dry_run=True, scopes=plan.results)
    kind = plan.kind
    by_scope = {r.scope: r for r in plan.results}
    job_info = None

    region_ids: list[str] = []
    vlm_ids: list[str] | None = None
    if 'region' in by_scope and kind != 'filter':
        region_ids = (await _plan_region(opensearch, request, kind))[1]
    if 'vlm' in by_scope:
        vlm_ids = (await _plan_vlm(opensearch, request, kind))[1]

    image_scopes = [s for s in _ORDER if s in by_scope and s in IMAGE_SCOPES]
    if image_scopes and plan.image_ids:
        if service_factory is None:
            raise ReprocessTargetsError('detect and embed need an ingest service')
        service = await service_factory()
        if len(plan.image_ids) > reprocess_sync_max():
            job_id = start_job(
                opensearch,
                service,
                request=request.model_dump(),
                scopes=image_scopes,
                image_ids=plan.image_ids,
            )
            job_info = read_job(job_id)
        else:
            done, _ = await process_images(
                opensearch, service, scopes=image_scopes, image_ids=plan.image_ids
            )
            for res in done:
                by_scope[res.scope].queued = res.queued
                by_scope[res.scope].failed = res.failed
                by_scope[res.scope].not_found = res.not_found
                by_scope[res.scope].detail = res.detail
                by_scope[res.scope].locked_skipped = res.locked_skipped

    if 'region' in by_scope:
        if kind == 'filter':
            assert request.targets.filter is not None
            by_scope['region'].queued = await apply_region_filter(
                opensearch, request.targets.filter, request.region_mode
            )
        else:
            by_scope['region'].queued = await apply_region_ids(
                opensearch, region_ids, request.region_mode
            )
    if 'vlm' in by_scope:
        if vlm_ids is None:
            assert request.targets.filter is not None
            hits = await scan_items(
                opensearch,
                {
                    'bool': {
                        'filter': [item_filter_query(request.targets.filter), vlm_class_clause()],
                        'must_not': [class_locked_clause()],
                    }
                },
                index=_cfg_indexes()[0],
                includes=['class_source'],
            )
            vlm_ids = [cid for cid, _ in hits]
        by_scope['vlm'].queued = await apply_vlm(opensearch, vlm_ids)
    return ReprocessResponse(dry_run=False, scopes=plan.results, job=job_info)


__all__ = [
    'ReprocessPlan',
    'apply_reprocess',
    'plan_reprocess',
    'redetect_image',
    'reembed_items',
    'resolve_image_ids',
]
