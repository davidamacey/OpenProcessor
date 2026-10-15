"""The auto-label run: cluster, auto-promote, then the VLM over what is left.

Called directly by the auto-label worker (through ``/start``'s job trigger) and
by the synchronous public route -- never a route itself, so the pinned-pack
arguments (``prompt_pack_revision`` / ``prompt_pack_resolved``) are unreachable
from an HTTP request.

1. (optional) Re-train FAISS clusters on every embedded item.
2. Auto-promote items in high-purity clusters (``cluster_propagation``).
3. Run the VLM over remaining unvalidated items with the open-vocabulary
   prompt -- high-confidence labels are auto-validated, ``__new__``
   proposals are flagged for the curator queue.

The result enumerates each stage's counts so the labeler dashboard can show
"this many items still need a human." Idempotent: safe to re-run.
"""

from __future__ import annotations

from typing import Any

from opensearchpy.exceptions import OpenSearchException

from src.config.curation import ITEM_EMBEDDING_FIELD
from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.services.curation.autolabel.embed_stage import run_embed_missing_stage
from src.services.curation.autolabel.health_snapshot import pipeline_health_snapshot
from src.services.curation.autolabel.run_params import (
    CLUSTER_SCOPED_SKIP,
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
from src.services.curation.cluster_purity import PROMOTE_MIN_MEMBERS, PROMOTE_MIN_PURITY
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

        await get_config_store().ensure_fresh(opensearch)
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


async def _normalize_stage(run: Run) -> dict[str, Any]:
    """Make ``cluster_id`` equal ``class_id`` for labeled items; the residual
    clusterer handles the rest."""
    from src.services.curation.autolabel.job import with_elapsed_tick

    start_cancellable_stage(run, 'cluster_id_normalize')
    try:
        from src.services.curation.clustering.id_normalize import force_cluster_id_equals_class_id

        return await with_elapsed_tick(
            run.progress, force_cluster_id_equals_class_id(run.opensearch)
        )
    except Exception as exc:
        logger.warning('pipeline_cluster_normalize_failed', error=str(exc))
        return {'status': 'error', 'error': str(exc)}


async def _cluster_residuals_stage(
    run: Run,
    *,
    reassign_only: bool,
    recluster_unvalidated: bool,
    clustering_method: str | None,
    gate_max_rank: int | None,
    gate_min_blur_ratio: float | None,
    n_clusters: int | None,
) -> dict[str, Any]:
    """AHC over the residual pool. ``reassign_only`` stream-assigns residuals against the
    persisted IVF centroids (no retrain); otherwise ``cluster_residuals`` dispatches via
    the ClusterMethod registry (a bad method becomes a stage error)."""
    from src.services.curation.autolabel.job import with_elapsed_tick
    from src.services.curation.clustering.cluster_geometry import cluster_geometry_stage

    start_cancellable_stage(run, 'cluster_residuals')
    try:
        if reassign_only:
            from src.services.curation.clustering.orchestrator import assign_only_residuals

            cluster_result = await assign_only_residuals(run.opensearch, progress=run.progress)
        else:
            from src.services.curation.clustering.orchestrator import cluster_residuals

            cluster_result = await cluster_residuals(
                run.opensearch,
                recluster_unvalidated=recluster_unvalidated,
                clustering_method=clustering_method,
                gate_max_rank=gate_max_rank,
                gate_min_blur_ratio=gate_min_blur_ratio,
                n_clusters=n_clusters,
                progress=run.progress,
            )
        result = dict(cluster_result)
    except Exception as exc:
        logger.warning('pipeline_cluster_residuals_failed', error=str(exc))
        result = {'status': 'error', 'error': str(exc)}
    # Every cluster (class ones too) gets centroid geometry.
    result['cluster_geometry'] = await with_elapsed_tick(
        run.progress, cluster_geometry_stage(run.opensearch)
    )
    return result


async def _auto_promote_stage(
    run: Run, *, run_auto_promote: bool, min_purity: float, min_members: int
) -> dict[str, Any]:
    from src.services.curation.autolabel.job import with_elapsed_tick
    from src.services.curation.clustering.auto_promote import gated_auto_promote

    start_cancellable_stage(run, 'auto_promote')
    if not run_auto_promote:
        # Disabled by default: the classifier+cluster-majority rule had no classifier
        # confidence floor and was contaminating class clusters by validating
        # low-confidence classifier predictions as long as they sat in a cluster whose
        # majority shared that class. Opt in via ?run_auto_promote=true.
        return {'skipped': True, 'reason': 'disabled by default'}
    if run.cluster_id is not None:
        return dict(CLUSTER_SCOPED_SKIP)
    try:
        await run.opensearch.indices.refresh(index=items_index())
    except OpenSearchException as exc:
        logger.warning('pipeline_pre_promote_refresh_failed', error=str(exc))
    promote = await with_elapsed_tick(
        run.progress,
        gated_auto_promote(
            run.opensearch, min_purity=min_purity, min_members=min_members, dry_run=False
        ),
    )
    return {
        'promoted': promote.get('promoted', 0),
        'skipped': promote.get('skipped', 0),
        'clusters_evaluated': len(promote.get('clusters', [])),
    }


async def _finish_without_vlm(run: Run, summary: dict[str, Any]) -> dict[str, Any]:
    """The VLM stage is default-off: the detection worker's combined call already writes
    ``class_id`` + ``vlm_verify_completed_at`` during region verification. Opt in with
    ``run_vlm=true`` to backfill the no-region cohort."""
    summary['stages']['vlm'] = skipped_vlm_stage()
    summary['stages']['cluster_id_normalize_post_vlm'] = {'skipped': True}
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
    train_clusters: bool = True,
    promote_min_purity: float = PROMOTE_MIN_PURITY,
    promote_min_members: int = PROMOTE_MIN_MEMBERS,
    vlm_batch_size: int = 32,
    vlm_concurrency: int = 8,
    max_vlm_crops: int = 0,
    classifier_confidence_skip_vlm: float = 0.80,
    clustering_method: str | None = None,
    run_vlm: bool = False,
    recluster_unvalidated: bool = False,
    reassign_only: bool = False,
    run_auto_promote: bool = False,
    gate_max_rank: int | None = None,
    gate_min_blur_ratio: float | None = None,
    n_clusters: int | None = None,
    class_id: int | None = None,
    cluster_id: int | None = None,
    detection_profile: str | None = None,
    prompt_pack: str | None = None,
    # Internal (job trigger only): the item filter scoping the embed and VLM stages.
    item_filter: dict[str, Any] | None = None,
    embed_missing: bool = False,
    vlm_scope: str | None = None,
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
        await resolve_run_vlm(opensearch, vlm, pack=None, acknowledge_external=acknowledge_external)
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

    # Pipeline order: classifier confident keeps its label; else -> VLM -> human.
    if train_clusters and cluster_id is not None:
        # Both stages rewrite cluster ids index-wide; a cluster-scoped job writes only to
        # the members it selected.
        for stage in ('cluster_id_normalize', 'cluster_residuals'):
            stages[stage] = dict(CLUSTER_SCOPED_SKIP)
    elif train_clusters:
        stages['cluster_id_normalize'] = await _normalize_stage(run)
        stages['cluster_residuals'] = await _cluster_residuals_stage(
            run,
            reassign_only=reassign_only,
            recluster_unvalidated=recluster_unvalidated,
            clustering_method=clustering_method,
            gate_max_rank=gate_max_rank,
            gate_min_blur_ratio=gate_min_blur_ratio,
            n_clusters=n_clusters,
        )
    stages['auto_promote'] = await _auto_promote_stage(
        run,
        run_auto_promote=run_auto_promote,
        min_purity=promote_min_purity,
        min_members=promote_min_members,
    )

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
    if result.post_normalize is not None:
        stages['cluster_id_normalize_post_vlm'] = result.post_normalize
    stages['vlm'] = result.stage
    summary['final'] = {'unvalidated': result.remaining, 'human_required': result.remaining}
    # The after-state and deltas, so the labeler can render "before/after" without re-querying.
    summary['after'] = await pipeline_health_snapshot(opensearch)
    baseline = summary.get('baseline') or {}
    after = summary['after']
    summary['delta'] = {
        k: int(after.get(k, 0)) - int(baseline.get(k, 0))
        for k in ('validated', 'unvalidated', 'has_class', 'cluster_id_mismatched')
        if k in after and k in baseline
    }
    return summary
