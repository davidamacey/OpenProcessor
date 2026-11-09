"""The auto-label run: embed what was stored without a vector, then the VLM over what is left.

Called directly by the auto-label worker (through ``/start``'s job trigger) and
by the synchronous public route -- never a route itself, so the pinned-pack
arguments (``prompt_pack_revision`` / ``prompt_pack_resolved``) are unreachable
from an HTTP request.

1. (optional) Embed the in-scope items stored without a vector.
2. Run the VLM over the unvalidated items with the open-vocabulary
   prompt -- high-confidence labels are auto-validated, ``__new__``
   proposals are flagged for the curator queue.

Clustering is not a stage of this run: the serving partition is built by the ``cluster_run``
job (``POST /cluster_partitions``) and kept current by the cluster refresh schedule.

The result enumerates each stage's counts so the labeler dashboard can show
"this many items still need a human." Idempotent: safe to re-run.
"""

from __future__ import annotations

from typing import Any

from src.config.curation import ITEM_EMBEDDING_FIELD
from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.services.curation.autolabel.embed_stage import run_embed_missing_stage
from src.services.curation.autolabel.health_snapshot import pipeline_health_snapshot
from src.services.curation.autolabel.run_params import (
    omitted_pack_is_store_active,
    reject_detection_profile,
    resolve_run_prompt_pack,
)
from src.services.curation.autolabel.run_state import Run, start_cancellable_stage
from src.services.curation.autolabel.run_vlm import resolve_run_vlm
from src.services.curation.autolabel.selection import (
    count_unvalidated_remaining,
    explain_empty_vlm_selection,
    resolve_vlm_selection,
    scroll_unvalidated,
    skipped_vlm_stage,
)
from src.services.curation.autolabel.vlm_stage import run_vlm_stage
from src.services.curation.class_write_guard import CLASS_GUARD_SOURCE_FIELDS, ClassWriteGuard
from src.services.curation.item_filter import ItemFilter


logger = get_logger(__name__)

# Fields the VLM sweep reads per unvalidated item.
# The class-state fields are the state the sweep decides on; the VLM
# write lands only if it is unchanged at write time.
VLM_SWEEP_SOURCE_FIELDS: tuple[str, ...] = (
    'crop_id',
    'image_path',
    'bbox_norm',
    ITEM_EMBEDDING_FIELD,
    *CLASS_GUARD_SOURCE_FIELDS,
)


async def _resolve_pack_pin(
    opensearch: Any,
    prompt_pack: str | None,
    prompt_pack_revision: int | None,
    *,
    prompt_pack_resolved: bool,
    prompt_pack_omitted: bool,
) -> tuple[str | None, int | None, bool]:
    """``(prompt_pack, prompt_pack_revision, prompt_pack_omitted)`` this run uses.

    ``/start`` already resolved the pin at request time when ``prompt_pack_resolved``;
    re-running ``resolve_run_prompt_pack`` on the bare name would silently re-pin to
    whatever is currently the latest saved revision, discarding the request's
    resolution (an explicit pin, or the active-pack-follow ``None`` revision).
    """
    if opensearch is not None and prompt_pack_resolved:
        # The auto-label worker runs in its own container, with a config-store snapshot
        # that starts empty and was refreshed nowhere on the job path: without this a
        # ``(name, revision)`` pin resolved at ``/start`` 404s here ("unknown prompt
        # pack") the moment the labeler reads it. Scoped to a ``/start``-resolved pin:
        # the synchronous route runs in-process and ``resolve_run_prompt_pack`` already
        # refreshes when it needs to.
        from src.services.config_store import get_config_store

        await get_config_store().ensure_fresh()
    if prompt_pack_resolved:
        return prompt_pack, prompt_pack_revision, prompt_pack_omitted
    # A direct synchronous call (never ``prompt_pack_resolved``, even with an explicit
    # name): resolve and validate now. "Was ``prompt_pack`` omitted" is taken from the
    # RAW param before the resolver overwrites it, the same signal ``/start`` passes.
    omitted = not isinstance(prompt_pack, str)
    name, revision = await resolve_run_prompt_pack(opensearch, prompt_pack)
    if omitted:
        # only the store-active case has TOCTOU draft risk.
        omitted = omitted_pack_is_store_active(name)
    return name, revision, omitted


async def _embed_missing_stage(run: Run) -> dict[str, Any]:
    """Lazy trigger: embed the in-scope items stored without a vector first."""
    start_cancellable_stage(run, 'embed_missing')
    return await run_embed_missing_stage(
        run.opensearch,
        class_id=run.class_id,
        cluster_id=run.cluster_id,
        item_filter=run.item_filter,
        progress=run.progress,
    )


async def _finish_without_vlm(run: Run, summary: dict[str, Any]) -> dict[str, Any]:
    """The VLM stage is default-off: the detection worker's combined call already writes
    ``class_id`` + ``vlm_verify_completed_at`` during region verification. Opt in with
    ``run_vlm=true`` to backfill the no-region cohort."""
    summary['stages']['vlm'] = skipped_vlm_stage()
    if run.progress is not None:
        run.progress.start_stage('vlm', total=0)
        run.progress.start_stage('finalize')
    summary['unvalidated_remaining'] = await count_unvalidated_remaining(
        run.opensearch, items_index(), run.class_id, run.cluster_id, run.scope_filter
    )
    summary['after'] = await pipeline_health_snapshot(run.opensearch)
    return summary


async def run_auto_label(
    opensearch: Any,
    vlm_batch_size: int = 32,
    vlm_concurrency: int = 8,
    max_vlm_crops: int = 0,
    classifier_confidence_skip_vlm: float = 0.80,
    run_vlm: bool = False,
    class_id: int | None = None,
    cluster_id: int | None = None,
    detection_profile: str | None = None,
    prompt_pack: str | None = None,
    # Internal (job trigger only): the item filter scoping the embed and VLM stages.
    item_filter: dict[str, Any] | None = None,
    embed_missing: bool = False,
    vlm_scope: str | None = None,
    vlm_attributes: bool = False,
    # Internal (never an HTTP param): ``/start`` resolves ``prompt_pack`` once at request
    # time and hands the resolved pin here.
    prompt_pack_revision: int | None = None,
    prompt_pack_resolved: bool = False,
    # The ORIGINAL request omitted ``prompt_pack``: resolve the labeler against "whatever
    # is active now", not the echoed name, which can go stale before the VLM stage runs.
    prompt_pack_omitted: bool = False,
    # The run's VLM endpoint. The public route passes the raw ``vlm`` / ``acknowledge_external``
    # (resolved and gated here); ``/start`` resolves and gates at request time and hands the
    # PINNED ``(name, revision)`` with ``vlm_resolved=True``.
    vlm: str | None = None,
    acknowledge_external: bool = False,
    vlm_endpoint: str | None = None,
    vlm_endpoint_revision: int | None = None,
    vlm_resolved: bool = False,
    progress: Any = None,
) -> dict[str, Any]:
    """Run the full auto-labeling chain end-to-end (see the module docstring)."""
    reject_detection_profile(detection_profile)
    if isinstance(vlm, str) and not vlm_resolved and not run_vlm:
        # Refused even when this run has no VLM stage, as /start does.
        await resolve_run_vlm(vlm, pack=None, acknowledge_external=acknowledge_external)
    prompt_pack, prompt_pack_revision, prompt_pack_omitted = await _resolve_pack_pin(
        opensearch,
        prompt_pack,
        prompt_pack_revision,
        prompt_pack_resolved=prompt_pack_resolved,
        prompt_pack_omitted=prompt_pack_omitted,
    )
    run = Run(
        opensearch=opensearch,
        progress=progress,
        class_id=class_id,
        cluster_id=cluster_id,
        item_filter=item_filter,
        scope_filter=ItemFilter(**(item_filter or {})),
    )
    summary: dict[str, Any] = {
        'stages': {},
        'class_id': class_id,
        'cluster_id': cluster_id,
        'prompt_pack': prompt_pack,
        'prompt_pack_revision': prompt_pack_revision,
        # Snapshot counts at entry for a real before/after.
        'baseline': await pipeline_health_snapshot(opensearch),
    }
    stages = summary['stages']
    if embed_missing:
        stages['embed_missing'] = await _embed_missing_stage(run)

    if not run_vlm:
        return await _finish_without_vlm(run, summary)

    # Selection (sweep vs cluster scope, VLM policy, caps): autolabel/selection.py.
    guard = ClassWriteGuard('vlm_pipeline')
    try:
        selection = await resolve_vlm_selection(
            opensearch,
            class_id=class_id,
            cluster_id=cluster_id,
            classifier_confidence_skip_vlm=classifier_confidence_skip_vlm,
            item_filter=run.scope_filter,
            max_vlm_crops=max_vlm_crops,
            scope_override=vlm_scope,
        )
        unvalidated_ids = await scroll_unvalidated(
            opensearch,
            index=items_index(),
            query=selection.query,
            source_fields=list(VLM_SWEEP_SOURCE_FIELDS),
            cap=selection.cap,
            guard=guard,
        )
    except Exception as exc:
        return {**summary, 'stages_error': f'fetch unvalidated failed: {exc}'}

    stages['unvalidated_after_promote'] = len(unvalidated_ids)
    if not unvalidated_ids:
        stages['vlm'] = skipped_vlm_stage(
            selection.empty_reason
            or await explain_empty_vlm_selection(
                opensearch, items_index(), class_id, cluster_id, run.scope_filter
            )
        )
        summary['final'] = {'unvalidated': 0, 'human_required': 0}
        return summary

    result = await run_vlm_stage(
        run,
        guard=guard,
        unvalidated_ids=unvalidated_ids,
        batch_size=vlm_batch_size,
        concurrency=vlm_concurrency,
        attributes=vlm_attributes,
        context_args={
            'prompt_pack': prompt_pack,
            'prompt_pack_revision': prompt_pack_revision,
            'prompt_pack_omitted': prompt_pack_omitted,
            'vlm': vlm,
            'acknowledge_external': acknowledge_external,
            'vlm_endpoint': vlm_endpoint,
            'vlm_endpoint_revision': vlm_endpoint_revision,
            'vlm_resolved': vlm_resolved,
        },
    )
    stages['vlm'] = result.stage
    summary['final'] = {'unvalidated': result.remaining, 'human_required': result.remaining}
    # The after-state and deltas, so the labeler can render "before/after" without re-querying.
    summary['after'] = await pipeline_health_snapshot(opensearch)
    baseline = summary.get('baseline') or {}
    after = summary['after']
    summary['delta'] = {
        k: int(after.get(k, 0)) - int(baseline.get(k, 0))
        for k in ('validated', 'unvalidated', 'has_class')
        if k in after and k in baseline
    }
    return summary
