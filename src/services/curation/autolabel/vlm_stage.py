"""The VLM stage of an auto-label run: label the unvalidated items in chunks and write the
answers with a concurrent human edit always winning."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch.crops import mget_crops
from src.config import get_region_fields
from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.services.curation.autolabel.run_params import labeler_resolution_args
from src.services.curation.autolabel.run_state import Run, start_stage
from src.services.curation.autolabel.run_vlm import job_endpoint
from src.services.curation.autolabel.selection import unvalidated_count_query
from src.services.curation.event_hub import publish_crop_classified
from src.services.labeling.run_labeler import get_vlm_labeler, resolve_pack, vlm_stamp
from src.services.labeling.vlm_factory import assert_may_connect


if TYPE_CHECKING:
    from src.services.curation.class_write_guard import ClassWriteGuard


logger = get_logger(__name__)

_ChunkResult = tuple[int, int, list[dict[str, Any]]]


@dataclass
class _VlmContext:
    """The labeler, the class vocabulary and the write guard one VLM stage uses."""

    labeler: Any
    guard: ClassWriteGuard
    class_names: list[str]
    name_to_id: dict[str, int]
    class_catalog: Any
    pack_stamp: str
    force_fit_bypass: dict[str, int]

    def resolve_class_name(self, raw: str, *, confidence: str | None = None) -> str | None:
        """Map a VLM reply to a registry class name (or None). Skips fuzzy/synonym
        resolution when ``confidence='low'`` -- see
        :func:`src.services.labeling.vlm_class_names.resolve_class_name`."""
        from src.services.labeling.vlm_class_names import resolve_class_name

        if confidence == 'low':
            self.force_fit_bypass['low_conf_skipped'] += 1
        else:
            self.force_fit_bypass['attempted'] += 1
        return resolve_class_name(raw, self.name_to_id, confidence=confidence)  # type: ignore[arg-type]


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
) -> _VlmContext:
    # Calls the labeler directly (no HTTP hop); ItemCrops are built here to chunk.
    from src.clients.curation_opensearch.registry import get_class_registry
    from src.services.curation.region_class import item_classes
    from src.services.labeling.vlm_class_names import format_class_catalog
    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    labelable = item_classes(get_class_registry().load().classes)
    # Render the registry as a grouped+described catalog so the VLM's prompt tells it what
    # each cryptic slug means visually -- a big quality lift over the bare CSV.
    class_dicts = [
        {'class_name': c.class_name, 'group': getattr(c, 'group', None)} for c in labelable
    ]
    # When the request omitted ``prompt_pack`` the labeler resolves against ``None``/``None``
    # -- always "whatever is active right now" -- never the echoed name/revision, which can
    # go stale between request time and here.
    labeler_pack, labeler_revision = labeler_resolution_args(
        prompt_pack, prompt_pack_revision, prompt_pack_omitted=prompt_pack_omitted
    )
    endpoint = await job_endpoint(
        run.opensearch,
        vlm=vlm,
        acknowledge_external=acknowledge_external,
        pack=resolve_pack(labeler_pack, labeler_revision),
        pinned_name=vlm_endpoint,
        pinned_revision=vlm_endpoint_revision,
        resolved=vlm_resolved,
    )
    if endpoint is not None:
        await assert_may_connect(endpoint)
    labeler = get_vlm_labeler(labeler_pack, labeler_revision, endpoint=endpoint)
    return _VlmContext(
        labeler=labeler,
        guard=guard,
        class_names=[c.class_name for c in labelable],
        name_to_id={c.class_name: c.class_id for c in labelable},
        class_catalog=format_class_catalog(class_dicts, labeler._pack),
        pack_stamp=prompt_pack_stamp(labeler._pack, revision=labeler_revision),
        # How many crops bypass the synonym/fuzzy force-fit because the VLM's confidence is
        # low: those route straight to the raw-label cluster pipeline.
        force_fit_bypass={'low_conf_skipped': 0, 'attempted': 0},
    )


def _chunk_crops(docs: dict[str, Any], ctx: _VlmContext) -> list[Any]:
    from src.services.curation.image_serving import (
        THUMBNAIL_CACHE,
        resolve_crop_root,
        resolve_safe_path,
    )
    from src.services.labeling.detector_hint import hinted_crop

    crops = []
    for crop_id, doc in docs.items():
        src = doc.get('_source') or {}
        image_path = src.get('image_path', '')
        bbox = src.get('bbox_norm')
        if not image_path or not bbox or len(bbox) != 4:
            continue
        try:
            safe = resolve_safe_path(image_path, resolve_crop_root(image_path))
            jpeg = THUMBNAIL_CACHE.get_or_compute(safe, tuple(bbox), size=224)
        except Exception as exc:
            logger.warning('pipeline_thumb_failed', crop_id=crop_id, error=str(exc))
            continue
        crops.append(hinted_crop(crop_id, jpeg, src, ctx.labeler._pack))
    return crops


def _updates_from_predictions(
    preds: list[Any], ctx: _VlmContext
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    from src.services.curation.vlm_class_attempt import prediction_class_update
    from src.services.detection.cascade_detect.sanity import class_provenance

    updates_by_id: dict[str, dict[str, Any]] = {}
    proposals: list[dict[str, Any]] = []
    now = datetime.now(UTC).isoformat()
    provenance = class_provenance(
        detector=ctx.labeler.identity.model,
        detector_version='1',
        labeler=ctx.labeler.identity.model,
        labeled_at=now,
    )
    for p in preds:
        # Always write the VLM's make/model/region_visible when reported, regardless of
        # which class-resolution path fires.
        extras: dict[str, Any] = {}
        if p.make:
            extras['vlm_item_make'] = p.make
        if p.model:
            extras['vlm_item_model'] = p.model
        if p.region_visible is not None:
            extras[get_region_fields().visible] = p.region_visible
        update, proposal = prediction_class_update(
            p,
            name_to_id=ctx.name_to_id,
            resolve=ctx.resolve_class_name,
            now=now,
            provenance=provenance,
            extras=extras,
            set_cluster=True,
        )
        if update is None:
            continue
        update['vlm_prompt_pack'] = ctx.pack_stamp
        update.update(vlm_stamp(ctx.labeler))
        if proposal is not None:
            proposals.append(proposal)
        updates_by_id[p.img_id] = update
    return updates_by_id, proposals


def _publish_classified(updates_by_id: dict[str, dict[str, Any]]) -> None:
    """Live UI updates per written crop."""
    for raw_id, doc in updates_by_id.items():
        if 'class_source' not in doc:
            continue  # empty answer: the class did not change
        try:
            publish_crop_classified(
                str(raw_id),
                class_id=doc.get('class_id'),
                class_name=doc.get('class_name'),
                class_source=doc.get('class_source', ''),
            )
        except Exception as exc:  # nosec B112 — advisory
            logger.debug('event_publish_failed', error=str(exc))


async def _write_updates(
    run: Run, ctx: _VlmContext, updates_by_id: dict[str, dict[str, Any]]
) -> None:
    """Worker-context bulk write via OCC: a concurrent human edit always wins, and
    ``class_id_history`` snapshots the prior assignment when a write changes the class."""
    from src.clients.occ import occ_skip_on_conflict_bulk
    from src.services.curation.vlm_class_attempt import with_class_snapshot

    def merge(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        # Only onto the class state the sweep selected on: a human write (or validation)
        # since the scroll wins.
        if not ctx.guard.allows(doc_id, current):
            return {}
        return with_class_snapshot(dict(updates_by_id[doc_id]), current, writer='vlm_pipeline')

    try:
        await occ_skip_on_conflict_bulk(
            run.opensearch,
            doc_ids=list(updates_by_id.keys()),
            merger=merge,
            index=items_index(),
            refresh=False,
            writer_id='vlm_pipeline',
        )
    except Exception as exc:
        logger.warning('pipeline_bulk_failed', error=str(exc))
    else:
        _publish_classified(updates_by_id)


async def _label_chunk(run: Run, ctx: _VlmContext, ids: list[str]) -> _ChunkResult:
    from src.services.labeling.detector_hint import DETECTOR_HINT_FIELDS

    # A single mget_crops call per chunk instead of per-crop opensearch.get.
    docs = await mget_crops(
        run.opensearch, ids, source_includes=['image_path', 'bbox_norm', *DETECTOR_HINT_FIELDS]
    )
    crops = _chunk_crops(docs, ctx)
    # No item classes yet (a fresh project, or only the region class): nothing to label as.
    if not crops or not ctx.class_names:
        return 0, 0, []
    preds = await ctx.labeler.label_or_propose_batch(
        crops, ctx.class_names, class_catalog=ctx.class_catalog
    )
    updates_by_id, proposals = _updates_from_predictions(preds, ctx)
    if updates_by_id:
        await _write_updates(run, ctx, updates_by_id)
    return len(preds), len(updates_by_id), proposals


async def _label_all(
    run: Run, ctx: _VlmContext, ids: list[str], *, batch_size: int, concurrency: int
) -> tuple[int, int, list[dict[str, Any]]]:
    """Label ``ids`` in chunks. The VLM is the dominant cost at scale (often hours), so
    progress is reported per chunk and an operator cancel is checked between chunks."""
    chunks = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]
    sem = asyncio.Semaphore(concurrency)
    start_stage(run, 'vlm', total=len(ids))

    async def gated(chunk: list[str]) -> _ChunkResult:
        async with sem:
            if run.progress is not None:
                run.progress.raise_if_cancelled()
            out = await _label_chunk(run, ctx, chunk)
            if run.progress is not None:
                run.progress.advance(len(chunk))
            return out

    results = await asyncio.gather(*[gated(c) for c in chunks])
    proposals: list[dict[str, Any]] = []
    for _, _, props in results:
        proposals.extend(props)
    return sum(r[0] for r in results), sum(r[1] for r in results), proposals


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
) -> VlmStageResult:
    ctx = await _build_vlm_context(run, guard, **context_args)
    predicted, updated, new_class_proposals = await _label_all(
        run, ctx, unvalidated_ids, batch_size=batch_size, concurrency=concurrency
    )
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
        'predicted': predicted,
        'updated': updated,
        'new_class_proposals': new_class_proposals[:50],
        'new_class_proposals_total': len(new_class_proposals),
        'force_fit_bypass': dict(ctx.force_fit_bypass),
    }
    return VlmStageResult(stage, post_normalize, remaining)
