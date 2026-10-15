"""The VLM stage of an auto-label run: label the unvalidated items in chunks and write the
answers with a concurrent human edit always winning."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch.crops import mget_crops
from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.services.curation.assist.chunk import suggested_name
from src.services.curation.assist.slot_write import RunMeta, build_slot, write_slots
from src.services.curation.autolabel.run_params import labeler_resolution_args
from src.services.curation.autolabel.run_state import Run, start_stage
from src.services.curation.autolabel.run_vlm import job_endpoint, require_item_attributes
from src.services.curation.autolabel.selection import unvalidated_count_query
from src.services.curation.vlm_class_batch import (
    ChunkResult,
    VlmContext,
    chunk_crops,
    class_context,
    updates_from_predictions,
    write_updates,
)
from src.services.labeling.run_labeler import get_vlm_labeler, resolve_pack
from src.services.labeling.vlm_factory import assert_may_connect
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from src.services.curation.class_write_guard import ClassWriteGuard


logger = get_logger(__name__)


async def _build_vlm_context(
    run: Run,
    guard: ClassWriteGuard,
    *,
    prompt_pack: str | None,
    prompt_pack_revision: int | None,
    prompt_pack_omitted: bool,
    vlm: str | None,
    acknowledge_external: bool,
    vlm_endpoint: str | None,
    vlm_endpoint_revision: int | None,
    vlm_resolved: bool,
    attributes: bool = False,
) -> tuple[VlmContext, RunMeta | None]:
    # When the request omitted ``prompt_pack`` the labeler resolves against ``None``/``None``
    # -- always "whatever is active right now" -- never the echoed name/revision, which can
    # go stale between request time and here.
    labeler_pack, labeler_revision = labeler_resolution_args(
        prompt_pack, prompt_pack_revision, prompt_pack_omitted=prompt_pack_omitted
    )
    pack = resolve_pack(labeler_pack, labeler_revision)
    if attributes:
        require_item_attributes(pack)
    endpoint = await job_endpoint(
        run.opensearch,
        vlm=vlm,
        acknowledge_external=acknowledge_external,
        pack=pack,
        pinned_name=vlm_endpoint,
        pinned_revision=vlm_endpoint_revision,
        resolved=vlm_resolved,
    )
    if endpoint is not None:
        await assert_may_connect(endpoint)
    labeler = get_vlm_labeler(labeler_pack, labeler_revision, endpoint=endpoint)
    ctx = class_context(labeler, guard, revision=labeler_revision)
    if not attributes:
        return ctx, None
    identity = labeler.identity
    meta = RunMeta(
        run_id=f'al_{uuid.uuid4().hex[:12]}',
        created_at=now_iso(),
        vlm_endpoint=identity.endpoint_ref,
        vlm_model=identity.model,
        prompt_pack=pack.name,
        prompt_pack_revision=labeler_revision,
    )
    return ctx, meta


async def _label_chunk(
    run: Run, ctx: VlmContext, ids: list[str], meta: RunMeta | None
) -> ChunkResult:
    from src.services.labeling.detector_hint import DETECTOR_HINT_FIELDS

    # A single mget_crops call per chunk instead of per-crop opensearch.get.
    docs = await mget_crops(
        run.opensearch, ids, source_includes=['image_path', 'bbox_norm', *DETECTOR_HINT_FIELDS]
    )
    crops = chunk_crops(docs, ctx)
    unreadable = len(docs) - len(crops)
    # No item classes yet (a fresh project, or only the region class): nothing to label as.
    if not crops or not ctx.class_names:
        return ChunkResult(0, 0, [], unreadable=unreadable)
    if meta is None:
        preds = await ctx.labeler.label_or_propose_batch(
            crops, ctx.class_names, class_catalog=ctx.class_catalog
        )
    else:
        preds = await ctx.labeler.assist_batch(
            crops, ctx.class_names, class_catalog=ctx.class_catalog
        )
    updates_by_id, proposals = updates_from_predictions(preds, ctx)
    if updates_by_id:
        await write_updates(run.opensearch, ctx, updates_by_id, writer_id='vlm_pipeline')
    if meta is not None:
        await write_slots(
            run.opensearch,
            {
                p.img_id: build_slot(
                    meta,
                    class_name=suggested_name(updates_by_id.get(p.img_id), p),
                    class_confidence=p.confidence,
                    attributes=p.attributes,
                )
                for p in preds
                if p.failure is None
            },
        )
    failed = sum(1 for p in preds if getattr(p, 'failure', None) == 'request_failed')
    return ChunkResult(len(preds), len(updates_by_id), proposals, failed, unreadable)


async def _label_all(
    run: Run,
    ctx: VlmContext,
    ids: list[str],
    *,
    batch_size: int,
    concurrency: int,
    meta: RunMeta | None,
) -> ChunkResult:
    """Label ``ids`` in chunks. The VLM is the dominant cost at scale (often hours), so
    progress is reported per chunk and an operator cancel is checked between chunks."""
    chunks = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]
    sem = asyncio.Semaphore(concurrency)
    start_stage(run, 'vlm', total=len(ids))

    async def gated(chunk: list[str]) -> ChunkResult:
        async with sem:
            if run.progress is not None:
                run.progress.raise_if_cancelled()
            out = await _label_chunk(run, ctx, chunk, meta)
            if run.progress is not None:
                run.progress.advance(len(chunk))
            return out

    results = await asyncio.gather(*[gated(c) for c in chunks])
    return ChunkResult(
        sum(r.predicted for r in results),
        sum(r.updated for r in results),
        [p for r in results for p in r.proposals],
        sum(r.failed for r in results),
        sum(r.unreadable for r in results),
    )


async def _remaining_unvalidated(run: Run) -> int:
    try:
        cnt_resp = await run.opensearch.count(
            index=items_index(),
            body={
                'query': unvalidated_count_query(
                    class_id=run.class_id,
                    cluster_id=run.cluster_id,
                    item_filter=run.scope_filter,
                )
            },
        )
        return int(cnt_resp.get('count', 0))
    except Exception:
        return -1


async def _post_vlm_normalize(run: Run, unvalidated_ids: list[str]) -> dict[str, Any] | None:
    """The VLM may have changed ``class_id`` without rewriting ``cluster_id``, which fragments
    the labeler view: one last pass makes ``cluster_id`` equal ``class_id`` wherever a class
    is set -- only on the selected items when the job is scoped."""
    from src.services.curation.autolabel.job import with_elapsed_tick
    from src.services.curation.clustering.id_normalize import force_cluster_id_equals_class_id

    start_stage(run, 'finalize')
    try:
        scoped = run.cluster_id is not None or run.class_id is not None
        return await with_elapsed_tick(
            run.progress,
            force_cluster_id_equals_class_id(
                run.opensearch, crop_ids=unvalidated_ids if scoped else None
            ),
        )
    except Exception as exc:
        logger.warning('pipeline_post_normalize_failed', error=str(exc))
        return None


@dataclass(frozen=True)
class VlmStageResult:
    """What the VLM stage reports back to the run."""

    stage: dict[str, Any]
    post_normalize: dict[str, Any] | None
    remaining: int


async def run_vlm_stage(
    run: Run,
    *,
    guard: ClassWriteGuard,
    unvalidated_ids: list[str],
    batch_size: int,
    concurrency: int,
    context_args: dict[str, Any],
    attributes: bool = False,
) -> VlmStageResult:
    ctx, meta = await _build_vlm_context(run, guard, attributes=attributes, **context_args)
    labeled = await _label_all(
        run, ctx, unvalidated_ids, batch_size=batch_size, concurrency=concurrency, meta=meta
    )
    if (
        unvalidated_ids
        and ctx.class_names
        and labeled.failed + labeled.unreadable >= len(unvalidated_ids)
    ):
        # Every selected crop failed: a "completed" run with predicted 0 would read as
        # "nothing to do". Fail closed with the counts.
        msg = (
            f'the VLM stage labeled nothing: {labeled.failed} request(s) failed and '
            f'{labeled.unreadable} crop image(s) could not be read, of {len(unvalidated_ids)} '
            'selected. Check that the VLM endpoint is reachable and the source images are '
            'mounted in the worker.'
        )
        raise RuntimeError(msg)
    try:
        await run.opensearch.indices.refresh(index=items_index())
    except OpenSearchException as exc:
        logger.warning('pipeline_refresh_failed', error=str(exc))
    post_normalize = await _post_vlm_normalize(run, unvalidated_ids)
    remaining = await _remaining_unvalidated(run)
    # How often the synonym/fuzzy force-fit was skipped: low-confidence replies fall through
    # to ``vlm_unmatched`` with the raw label kept for downstream clustering.
    logger.info(
        'pipeline_force_fit_bypass',
        attempted=ctx.force_fit_bypass['attempted'],
        low_conf_skipped=ctx.force_fit_bypass['low_conf_skipped'],
    )
    stage = {
        'predicted': labeled.predicted,
        'updated': labeled.updated,
        'failed': labeled.failed,
        'unreadable': labeled.unreadable,
        'new_class_proposals': labeled.proposals[:50],
        'new_class_proposals_total': len(labeled.proposals),
        'force_fit_bypass': dict(ctx.force_fit_bypass),
    }
    return VlmStageResult(stage, post_normalize, remaining)
