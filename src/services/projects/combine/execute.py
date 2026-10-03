"""The combine job body (projects plan section 6): registry, images and items
page by page with a durable per-chunk mark, the holdout record, then a
terminal status. The target project's own status (``building`` -> ``active``
or ``failed``) is the caller's (:func:`~src.services.projects.lifecycle.
finish_building`), handed in as ``settle`` and awaited BEFORE the job's
terminal status is written, so a job never reads done while its target is
still ``building``.

Resume re-reads the persisted plan (``request.json`` + ``mapping.json``), never
the sources' current classes, so a resumed run copies what the preview showed.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch import ClassRegistry
from src.config.project_context import bind_project
from src.config.region_fields import get_region_fields
from src.core.logging import get_logger
from src.services.curation.dataset_import.mapping import MapTarget, norm_class_name
from src.services.curation.dataset_import.project_source import class_names_by_id, iter_source_pages
from src.services.curation.dataset_import.store import now_iso
from src.services.curation.file_job import heartbeat_ticker
from src.services.curation.next_steps import recluster_items
from src.services.projects.combine import holdout
from src.services.projects.combine.image_copy import CopyContext, copy_page
from src.services.projects.combine.mapping import CombineMapping
from src.services.projects.combine.models import CombineRequest
from src.services.projects.combine.plan import Analysis, duplicates_from_wire, duplicates_wire


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.config.projects import ProjectRecord
    from src.services.curation.dataset_import.store import ImportStore
    from src.services.curation.file_job import FileJob

logger = get_logger(__name__)


class FencedError(Exception):
    """Another worker has claimed this job since this one started."""


def ensure_owner(job: FileJob, claim: str) -> None:
    """Raise :class:`FencedError` unless ``claim`` is still the job's claim.

    Every worker writes a fresh claim into the job state when it starts
    (:func:`run_combine`) and re-checks it at each chunk boundary, so a worker
    that stalled past the stale heartbeat window and was taken over stops at
    its next boundary instead of writing alongside its successor. The check
    and the write after it are not atomic across processes: a fenced worker
    can still finish the chunk it is in (chunk writes are deterministic-id
    ``index`` ops, so that is repeated work, not duplicates).
    """
    if job.read().get('claim') != claim:
        raise FencedError(claim)


@dataclass
class Plan:
    """What a job runs: persisted at start, reloaded on resume."""

    request: CombineRequest
    mapping: CombineMapping
    duplicates: dict[tuple[int, str], tuple[int, str]]
    originals: dict[tuple[int, str], tuple[str, str]]
    images_total: int


def persist_plan(store: ImportStore, analysis: Analysis, preview_sha: str) -> None:
    prints = {(f.source_index, f.image_id): f for f in analysis.fingerprints}
    needed = set(analysis.duplicates.values())
    store.write_request({**analysis.request.model_dump(mode='json'), 'preview_sha': preview_sha})
    store.write_mapping(
        {
            'target_classes': analysis.mapping.target_classes,
            'per_source': {
                slug: {
                    name: {'kind': t.kind, 'class_name': t.class_name} for name, t in table.items()
                }
                for slug, table in analysis.mapping.per_source.items()
            },
            'duplicates': duplicates_wire(analysis.duplicates),
            'originals': {
                f'{i}:{image}': [prints[(i, image)].path, prints[(i, image)].imohash]
                for i, image in needed
            },
            'images_total': sum(s.images for s in analysis.stats),
            'target_class_ids': {},
        }
    )


def load_plan(store: ImportStore) -> Plan:
    raw = store.read_request()
    raw.pop('preview_sha', None)
    request = CombineRequest.model_validate(raw)
    saved = store.read_mapping()
    mapping = CombineMapping(
        target_classes=list(saved['target_classes']),
        per_source={
            slug: {
                name: MapTarget(kind=row['kind'], class_name=row.get('class_name'))
                for name, row in table.items()
            }
            for slug, table in saved['per_source'].items()
        },
    )
    originals = {}
    for key, (path, imohash) in saved['originals'].items():
        i, _, image = key.partition(':')
        originals[(int(i), image)] = (path, imohash)
    return Plan(
        request=request,
        mapping=mapping,
        duplicates=duplicates_from_wire(saved['duplicates']),
        originals=originals,
        images_total=int(saved.get('images_total') or 0),
    )


def build_target_registry(target: ProjectRecord, names: list[str]) -> dict[str, int]:
    """Create the target's classes in order (a name already there keeps its
    id, so a resume is a no-op) and return ``name -> class_id``."""
    registry = ClassRegistry(path=target.resources.class_registry_path)
    have = {
        norm_class_name(c.class_name): c.class_id
        for c in registry.load().classes
        if not c.deprecated
    }
    ids: dict[str, int] = {}
    for name in names:
        norm = norm_class_name(name)
        ids[name] = have[norm] if norm in have else registry.add_class(name)
    return ids


def page_size() -> int:
    """Images per chunk (``OP_COMBINE_PAGE_SIZE``): the unit a resume redoes."""
    raw = os.environ.get('OP_COMBINE_PAGE_SIZE', '')
    return int(raw) if raw.isdigit() and int(raw) > 0 else 200


def _resume_points(done: dict[int, dict[str, Any]]) -> dict[int, str]:
    last: dict[int, str] = {}
    for report in done.values():
        source = report.get('source')
        if isinstance(source, int):
            last[source] = max(last.get(source, ''), str(report.get('last') or ''))
    return last


def _sum_reports(done: dict[int, dict[str, Any]]) -> Counter[str]:
    total: Counter[str] = Counter()
    for report in done.values():
        total.update({k: v for k, v in report.items() if isinstance(v, int) and k != 'source'})
    return total


def _publish(job_id: str, target: str, state: dict[str, Any]) -> None:
    from src.services.curation.event_hub import publish_global_event

    with contextlib.suppress(Exception):
        publish_global_event(
            'combine.progress',
            target=target,
            job_id=job_id,
            phase=state.get('phase'),
            done=state.get('done'),
            total=state.get('total'),
            status=state.get('status'),
        )


async def run_combine(
    client: Any,
    *,
    store: ImportStore,
    plan: Plan,
    sources: list[ProjectRecord],
    target: ProjectRecord,
    embedding_dim: int,
    settle: Callable[[bool], Awaitable[None]],
) -> bool:
    """Run (or resume) one combine to a terminal job status. ``settle(ok)``
    moves the target out of ``building`` and runs before that status is
    written (a cancelled or fenced run never settles). Never raises:
    a failure is recorded as ``failed`` with its error. ``False`` when another
    worker claimed the job meanwhile (:class:`FencedError`): this one wrote
    nothing after that and its caller must not settle the target."""
    job = store.job
    job_id = store.import_id
    claim = uuid.uuid4().hex
    job.touch_heartbeat()
    ticker = asyncio.create_task(heartbeat_ticker(job))
    try:
        done = store.chunks_done()
        job.update(
            status='running', phase='registry', total=plan.images_total, error=None, claim=claim
        )
        ids = build_target_registry(target, plan.mapping.target_classes)
        saved = store.read_mapping()
        saved['target_class_ids'] = ids
        store.write_mapping(saved)
        ctx = CopyContext(
            client=client,
            job_id=job_id,
            request=plan.request,
            sources=sources,
            target=target,
            mapping=plan.mapping,
            target_ids=ids,
            names=[class_names_by_id(r) for r in sources],
            fields=get_region_fields(),
            embedding_dim=embedding_dim,
            now=now_iso(),
            duplicates=plan.duplicates,
            originals=plan.originals,
        )
        if not await _copy_all(ctx, store, plan, done, claim):
            await _finish(ctx, store, claim, settle)
    except FencedError:
        logger.warning('combine_worker_fenced', job_id=job_id, claim=claim)
        return False
    except Exception as exc:
        logger.error('combine_failed', job_id=job_id, error=str(exc))
        try:
            ensure_owner(job, claim)
        except FencedError:
            return False
        error = str(exc)[:300]
        if not isinstance(exc, SettleError):
            try:
                await settle(False)
            except Exception as settle_exc:
                logger.error('combine_settle_failed', job_id=job_id, error=str(settle_exc))
                error = f'{error}; could not fail the target: {settle_exc}'[:300]
        job.update(status='failed', error=error, finished_at=now_iso())
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker
        _publish(job_id, target.slug, job.read())
    return True


async def _copy_all(
    ctx: CopyContext, store: ImportStore, plan: Plan, done: dict[int, dict[str, Any]], claim: str
) -> bool:
    """Every unfinished chunk; ``True`` when the job was cancelled."""
    job = store.job
    job.update(phase='images')
    resume = _resume_points(done)
    chunk = max(done, default=-1) + 1
    report = _sum_reports(done)
    for source_index, (source, record) in enumerate(
        zip(plan.request.sources, ctx.sources, strict=True)
    ):
        pages = iter_source_pages(
            ctx.client,
            record,
            validated_only=source.include.label_states == 'validated_only',
            with_vectors=True,
            page_size=page_size(),
            start_after=resume.get(source_index),
        )
        async for page in pages:
            ensure_owner(job, claim)
            if job.cancel_requested():
                job.update(status='cancelled', finished_at=now_iso())
                return True
            counts = await copy_page(ctx, source_index, page)
            ensure_owner(job, claim)
            store.mark_chunk_done(
                chunk, {'source': source_index, 'last': page[-1].image_id, **counts}
            )
            chunk += 1
            report.update(counts)
            seen = report['images_copied'] + report['images_duplicate']
            job.update(done=seen, report=dict(report), updated_at=now_iso())
            _publish(store.import_id, ctx.target.slug, job.read())
    return False


class SettleError(Exception):
    """The target could not leave ``building`` after a complete copy."""


async def _finish(
    ctx: CopyContext, store: ImportStore, claim: str, settle: Callable[[bool], Awaitable[None]]
) -> None:
    job = store.job
    ensure_owner(job, claim)
    job.update(phase='holdout')
    with bind_project(ctx.target):
        await ctx.client.indices.refresh(index=ctx.items_index)
    mode = ctx.request.holdout
    extra: dict[str, Any] = {}
    if mode == 'preserve_union':
        extra = await holdout.record_union(ctx.client, ctx.target, ctx.items_index, store.import_id)
    elif mode == 'recompute':
        extra = await holdout.recompute(ctx.client, ctx.target, ctx.items_index, store.import_id)
    report = {**_sum_reports(store.chunks_done()), **extra}
    next_steps = [recluster_items()]
    ensure_owner(job, claim)  # the holdout step is long; a takeover may have landed in it
    try:
        await settle(True)
    except Exception as exc:
        raise SettleError(f'could not finish the target: {exc}') from exc
    job.update(
        status='completed',
        phase='done',
        report=report,
        next_steps=next_steps,
        finished_at=now_iso(),
    )


__all__ = [
    'FencedError',
    'Plan',
    'SettleError',
    'build_target_registry',
    'ensure_owner',
    'load_plan',
    'persist_plan',
    'run_combine',
]
