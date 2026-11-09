"""The combine job body (projects plan section 6): registry, images and items
page by page with a durable checkpoint per chunk, the holdout record, then the
result. The target project's own status (``building`` -> ``active`` or ``failed``)
is the caller's (:func:`~src.services.projects.lifecycle.finish_building`), handed
in as ``settle`` and awaited BEFORE the job's terminal status is written, so a job
never reads done while its target is still ``building``.

The job runs in the target's scope (``jobs.project_id`` is the target); the sources are
read through the project binding of each source in turn. Resume re-reads the persisted
plan (the job parameters and the run's ``class_mapping``), never the sources' current
classes, so a resumed run copies what the preview showed; the cursor is the job
checkpoint: the last image id per source and the counters so far.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config.env import env_int
from src.config.project_context import bind_project
from src.config.region_fields import get_region_fields
from src.core.logging import get_logger
from src.services.classes.registry import get_class_registry
from src.services.curation.dataset_import.mapping import MapTarget
from src.services.curation.dataset_import.project_source import class_names_by_id, iter_source_pages
from src.services.curation.next_steps import recluster_items
from src.services.projects.combine import holdout
from src.services.projects.combine.image_copy import CopyContext, copy_page
from src.services.projects.combine.mapping import CombineMapping
from src.services.projects.combine.models import CombineRequest
from src.services.projects.combine.plan import Analysis, duplicates_from_wire, duplicates_wire
from src.services.projects.combine.spaces import plan_space_copy
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from src.config.projects import ProjectRecord
    from src.services.projects.combine.image_copy import PendingEnqueue
    from src.tasks.runtime.context import JobContext

logger = get_logger(__name__)


class LeaseLostError(Exception):
    """The job lost its lease (another worker owns it now): stop without writing."""


@dataclass
class Plan:
    """What a job runs: persisted at start, reloaded on resume."""

    request: CombineRequest
    mapping: CombineMapping
    duplicates: dict[tuple[int, str], tuple[int, str]]
    originals: dict[tuple[int, str], tuple[str, str]]
    images_total: int


def plan_documents(analysis: Analysis, preview_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(job params, run class_mapping document)`` of a previewed combine."""
    prints = {(f.source_index, f.image_id): f for f in analysis.fingerprints}
    needed = set(analysis.duplicates.values())
    params = {**analysis.request.model_dump(mode='json'), 'preview_sha': preview_sha}
    document = {
        'target_classes': analysis.mapping.target_classes,
        'per_source': {
            slug: {name: {'kind': t.kind, 'class_name': t.class_name} for name, t in table.items()}
            for slug, table in analysis.mapping.per_source.items()
        },
        'duplicates': duplicates_wire(analysis.duplicates),
        'originals': {
            f'{i}:{image}': [prints[(i, image)].path, prints[(i, image)].imohash]
            for i, image in needed
        },
        'images_total': sum(s.images for s in analysis.stats),
    }
    return params, document


def load_plan(params: Mapping[str, Any], document: Mapping[str, Any]) -> Plan:
    raw = dict(params)
    raw.pop('preview_sha', None)
    request = CombineRequest.model_validate(raw)
    mapping = CombineMapping(
        target_classes=list(document['target_classes']),
        per_source={
            slug: {
                name: MapTarget(kind=row['kind'], class_name=row.get('class_name'))
                for name, row in table.items()
            }
            for slug, table in document['per_source'].items()
        },
    )
    originals = {}
    for key, (path, imohash) in document['originals'].items():
        i, _, image = key.partition(':')
        originals[(int(i), image)] = (path, imohash)
    return Plan(
        request=request,
        mapping=mapping,
        duplicates=duplicates_from_wire(document['duplicates']),
        originals=originals,
        images_total=int(document.get('images_total') or 0),
    )


async def build_target_registry(target: ProjectRecord, names: list[str]) -> dict[str, int]:
    """Create the target's classes in order (a name already there keeps its
    id, so a resume is a no-op) and return ``name -> class_id``."""
    ids: dict[str, int] = {}
    with bind_project(target):
        registry = get_class_registry()
        for name in names:
            ids[name], _created = await registry.ensure_class(name)
    return ids


def page_size() -> int:
    """Images per chunk (``OP_COMBINE_PAGE_SIZE``): the unit a resume redoes."""
    return env_int('OP_COMBINE_PAGE_SIZE', 200, minimum=1)


def _sum_counts(counts: Mapping[str, Any] | None) -> Counter[str]:
    return Counter({k: v for k, v in (counts or {}).items() if isinstance(v, int)})


def _resume_points(cursor: Mapping[str, Any] | None) -> dict[int, str]:
    last = (cursor or {}).get('last') or {}
    return {int(source): str(image) for source, image in last.items()}


def _check_lease(ctx: JobContext) -> None:
    if ctx.lost():
        raise LeaseLostError(str(ctx.job_id))


async def run_combine(
    ctx: JobContext,
    client: Any,
    *,
    plan: Plan,
    sources: list[ProjectRecord],
    target: ProjectRecord,
    settle: Callable[[bool], Awaitable[None]],
    enqueue_pending: PendingEnqueue,
) -> dict[str, Any] | None:
    """Run (or resume) one combine; the result is ``{report, next_steps}``.

    ``settle(ok)`` moves the target out of ``building`` and runs before the job ends. A failure
    settles the target as failed and re-raises, so the wrapper records the job as ``failed``
    with the error; a cancelled run settles it failed as well (cancel is terminal) and returns
    ``None``. A run that lost its lease writes nothing more.
    """
    try:
        await ctx.progress(0, plan.images_total, 'registry')
        ids = await build_target_registry(target, plan.mapping.target_classes)
        copy_ctx = CopyContext(
            client=client,
            job_id=str(ctx.job_id),
            request=plan.request,
            sources=sources,
            target=target,
            mapping=plan.mapping,
            target_ids=ids,
            names=[await class_names_by_id(r) for r in sources],
            fields=get_region_fields(),
            spaces=[await plan_space_copy(r, target) for r in sources],
            now=now_iso(),
            duplicates=plan.duplicates,
            originals=plan.originals,
            enqueue_pending=enqueue_pending,
        )
        cancelled, counts = await _copy_all(ctx, copy_ctx, plan)
        if cancelled:
            await _settle_quietly(settle, False)
            return None
        return await _finish(ctx, copy_ctx, settle, counts, plan.images_total)
    except LeaseLostError:
        logger.warning('combine_lease_lost', job_id=str(ctx.job_id))
        return None
    except SettleError:
        raise
    except Exception:
        _check_lease(ctx)
        await _settle_quietly(settle, False)
        raise


async def _settle_quietly(settle: Callable[[bool], Awaitable[None]], ok: bool) -> None:
    try:
        await settle(ok)
    except Exception as exc:
        logger.error('combine_settle_failed', error=str(exc))


async def _copy_all(
    ctx: JobContext, copy_ctx: CopyContext, plan: Plan
) -> tuple[bool, Counter[str]]:
    """Every unfinished chunk: ``(cancelled, counters)``."""
    cursor: Mapping[str, Any] | None = ctx.resume_from
    resume = _resume_points(cursor)
    last: dict[str, str] = {str(k): v for k, v in resume.items()}
    chunk = int((cursor or {}).get('chunk', 0))
    report = _sum_counts((cursor or {}).get('counts'))
    for source_index, (source, record) in enumerate(
        zip(plan.request.sources, copy_ctx.sources, strict=True)
    ):
        pages = iter_source_pages(
            copy_ctx.client,
            record,
            validated_only=source.include.label_states == 'validated_only',
            with_vectors=True,
            page_size=page_size(),
            start_after=resume.get(source_index),
        )
        async for page in pages:
            _check_lease(ctx)
            if ctx.cancelled():
                return True, report
            counts = await copy_page(copy_ctx, source_index, page)
            report.update(counts)
            last[str(source_index)] = page[-1].image_id
            chunk += 1
            await ctx.checkpoint({'chunk': chunk, 'last': dict(last), 'counts': dict(report)})
            seen = report['images_copied'] + report['images_duplicate']
            await ctx.progress(seen, plan.images_total, 'images', dict(report))
    return False, report


class SettleError(Exception):
    """The target could not leave ``building`` after a complete copy."""


async def _finish(
    ctx: JobContext,
    copy_ctx: CopyContext,
    settle: Callable[[bool], Awaitable[None]],
    counts: Counter[str],
    total: int,
) -> dict[str, Any]:
    _check_lease(ctx)
    seen = counts['images_copied'] + counts['images_duplicate']
    await ctx.progress(seen, total, 'holdout', dict(counts), force=True)
    with bind_project(copy_ctx.target):
        await copy_ctx.client.indices.refresh(index=copy_ctx.items_index)
    mode = copy_ctx.request.holdout
    job_id = str(ctx.job_id)
    extra: dict[str, Any] = {}
    if mode == 'preserve_union':
        extra = await holdout.record_union(
            copy_ctx.client, copy_ctx.target, copy_ctx.items_index, job_id
        )
    elif mode == 'recompute':
        extra = await holdout.recompute(
            copy_ctx.client, copy_ctx.target, copy_ctx.items_index, job_id
        )
    report = {**counts, **extra}
    _check_lease(ctx)  # the holdout step is long; a takeover may have landed in it
    try:
        await settle(True)
    except Exception as exc:
        raise SettleError(f'could not finish the target: {exc}') from exc
    return {'report': report, 'next_steps': [recluster_items()]}


__all__ = [
    'LeaseLostError',
    'Plan',
    'SettleError',
    'build_target_registry',
    'load_plan',
    'plan_documents',
    'run_combine',
]
