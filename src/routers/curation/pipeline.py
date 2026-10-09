"""Curation auto-label router (SSE: pipeline_events; status/cancel: pipeline_control)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Query

from src.clients.curation_opensearch.crops import mget_crops
from src.config import get_region_fields
from src.config.curation import ITEM_EMBEDDING_FIELD
from src.routers.curation._common import (
    OpenSearchDep,
    _now_iso,
    get_class_registry,
    items_index,
    logger,
)
from src.routers.curation.pipeline_params import (
    AUTO_PROMOTE_DESC as _AUTO_PROMOTE_DESC,
    CLASS_ID_DESC as _CLASS_ID_DESC,
    CLUSTER_ID_DESC as _CLUSTER_ID_DESC,
    CLUSTER_SCOPED_SKIP,
    PROMPT_PACK_DESC as _PROMPT_PACK_DESC,
    REASSIGN_ONLY_DESC as _REASSIGN_ONLY_DESC,
    labeler_resolution_args,
    omitted_pack_is_store_active,
    reject_detection_profile,
    resolve_run_prompt_pack,
)
from src.routers.curation.pipeline_vlm import job_endpoint, resolve_run_vlm
from src.routers.curation.vlm import _get_vlm_labeler, _resolve_pack, _vlm_stamp
from src.services.curation.autolabel.embed_stage import run_embed_missing_stage
from src.services.curation.autolabel.selection import (
    count_unvalidated_remaining,
    explain_empty_vlm_selection,
    resolve_vlm_selection,
    scroll_unvalidated,
    skipped_vlm_stage,
    unvalidated_count_query,
)
from src.services.curation.class_write_guard import CLASS_GUARD_SOURCE_FIELDS, ClassWriteGuard
from src.services.curation.cluster_purity import PROMOTE_MIN_MEMBERS, PROMOTE_MIN_PURITY
from src.services.curation.event_hub import publish_crop_classified
from src.services.curation.item_filter import ItemFilter
from src.services.labeling.vlm_factory import assert_may_connect


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


# POST /pipeline/auto_label/start (the job-dispatch route) lives in
# pipeline_start.py (700-LOC ratchet split, same pattern as
# _region_profile_clone.py). It imports `_run_auto_label` back from this
# module at call time (not at import time, to dodge the circular import).
from src.routers.curation import pipeline_start  # noqa: E402,F401


async def _run_auto_label(
    opensearch: OpenSearchDep,
    # Annotated defaults (not `= Query(...)`): the auto-label worker calls
    # this function directly with only the args in its trigger file, and a
    # `Query(...)` default would leak in as a FieldInfo object.
    train_clusters: Annotated[
        bool, Query(description='Re-train clusters on the items index before promote/VLM.')
    ] = True,
    promote_min_purity: Annotated[float, Query(ge=0.5, le=1.0)] = PROMOTE_MIN_PURITY,
    promote_min_members: Annotated[int, Query(ge=2, le=1000)] = PROMOTE_MIN_MEMBERS,
    vlm_batch_size: Annotated[int, Query(ge=4, le=64)] = 32,
    vlm_concurrency: Annotated[int, Query(ge=1, le=128)] = 8,
    max_vlm_crops: Annotated[int, Query(ge=0, le=100000, description='0 = all in scope')] = 0,
    classifier_confidence_skip_vlm: Annotated[
        float,
        Query(
            ge=0.0,
            le=1.0,
            description=(
                'Skip the VLM for classifier-labeled items at or above this confidence '
                '(global sweep only; a cluster-scoped run labels every unvalidated member).'
            ),
        ),
    ] = 0.80,
    clustering_method: Annotated[
        str | None,
        Query(description='Residual-pool clusterer id from GET /methods (axis=cluster).'),
    ] = None,
    run_vlm: Annotated[bool, Query(description='Run the VLM stage. See /start.')] = False,
    recluster_unvalidated: Annotated[bool, Query(description='Merge candidate clusters.')] = False,
    reassign_only: Annotated[bool, Query(description=_REASSIGN_ONLY_DESC)] = False,
    run_auto_promote: Annotated[bool, Query(description=_AUTO_PROMOTE_DESC)] = False,
    # Cluster scope (primary-subject gate) — see /pipeline/auto_label/start.
    gate_max_rank: Annotated[int | None, Query(ge=1)] = None,
    gate_min_blur_ratio: Annotated[float | None, Query(ge=0.0)] = None,
    n_clusters: Annotated[int | None, Query(ge=2, le=4096)] = None,
    class_id: Annotated[int | None, Query(description=_CLASS_ID_DESC)] = None,
    cluster_id: Annotated[int | None, Query(description=_CLUSTER_ID_DESC)] = None,
    detection_profile: Annotated[str | None, Query(include_in_schema=False)] = None,
    prompt_pack: Annotated[str | None, Query(description=_PROMPT_PACK_DESC)] = None,
    # Internal (job trigger only): the item filter scoping the embed and VLM stages.
    item_filter: dict[str, Any] | None = None,
    embed_missing: bool = False,
    vlm_scope: str | None = None,
    # Internal (never an HTTP param; the public route forces the defaults):
    # `/start` resolves `prompt_pack` once at request time and hands the
    # resolved pin here (R4-3). `prompt_pack_revision` is that pin.
    prompt_pack_revision: int | None = None,
    # `/start` already ran `resolve_run_prompt_pack` (R5-2): do not re-pin.
    prompt_pack_resolved: bool = False,
    # The ORIGINAL request omitted `prompt_pack` (R6-1b): resolve the
    # labeler against "whatever is active now", not the echoed name, which
    # can go stale before the VLM stage runs.
    prompt_pack_omitted: bool = False,
    # W9: the run's VLM endpoint. The public route passes the raw `vlm` /
    # `acknowledge_external` (resolved and gated here); `/start` resolves
    # and gates at request time and hands the PINNED `(name, revision)`
    # with `vlm_resolved=True`, which the public route can never set.
    vlm: str | None = None,
    acknowledge_external: bool = False,
    vlm_endpoint: str | None = None,
    vlm_endpoint_revision: int | None = None,
    vlm_resolved: bool = False,
    progress: Any = None,
) -> dict[str, Any]:
    """Run the full auto-labeling chain end-to-end:

    1. (optional) Re-train FAISS clusters on every embedded item.
    2. Auto-promote items in high-purity clusters (``cluster_propagation``).
    3. Run the VLM over remaining unvalidated items with the open-vocabulary
       prompt — high-confidence labels are auto-validated, ``__new__``
       proposals are flagged for the curator queue.

    Output enumerates each stage's counts so the labeler dashboard can show
    "this many items still need a human." Idempotent: safe to re-run.

    Internal implementation, called directly by the auto-label worker
    (via ``/start``'s job trigger) and by the public route below --
    never a route itself (R6-m1 fix): `prompt_pack_revision` /
    `prompt_pack_resolved` must never be reachable from an HTTP request.
    """
    import asyncio as _asyncio

    from src.routers.curation.pipeline_health import pipeline_health_snapshot
    from src.services.curation.autolabel.job import with_elapsed_tick
    from src.services.curation.clustering.auto_promote import gated_auto_promote
    from src.services.curation.image_serving import THUMBNAIL_CACHE
    from src.services.labeling.vlm_models import ItemCrop

    reject_detection_profile(detection_profile)
    if isinstance(vlm, str) and not vlm_resolved and not run_vlm:
        # Refused even when this run has no VLM stage, as /start does.
        await resolve_run_vlm(opensearch, vlm, pack=None, acknowledge_external=acknowledge_external)
    if opensearch is not None and prompt_pack_resolved:
        # R6-1a fix (Blocker, W3/W4 round-6 review): auto_label_worker runs
        # this in its OWN container/process, with its own config-store
        # snapshot that starts empty and was refreshed nowhere on the job
        # path -- `/start`'s own `ensure_fresh` runs in the yolo-api
        # process, and a different process's in-memory snapshot does not
        # inherit that. Without this, a `(name, revision)` pin resolved at
        # `/start` time 404s here every time ("unknown prompt pack") the
        # moment `_get_vlm_labeler` reads it. Scoped to `prompt_pack_
        # resolved` (true only for a job trigger carrying a `/start`-
        # resolved pin) rather than unconditional: the synchronous public
        # route always passes `prompt_pack_resolved=False` and runs
        # entirely in-process (no cross-process staleness to guard
        # against), and `resolve_run_prompt_pack` below already refreshes
        # when it needs to. An unconditional refresh here was proven, by a
        # genuine `test_cross_project_leak.py` regression, to add a real
        # (if TTL-gated) extra OpenSearch call to a route that never
        # needed one -- not worth it for zero correctness gain there.
        from src.services.config_store import get_config_store

        await get_config_store().ensure_fresh(opensearch)
    if not prompt_pack_resolved:
        # No prior resolution handed in -- a direct synchronous call
        # (`prompt_pack_resolved` is never set for that route, even when
        # the caller passes an explicit `prompt_pack` name). Resolve (and
        # validate) it now, same as before this fix.
        #
        # R5-2 fix (W3/W4 round-5 review, Blocker): this used to key off
        # `prompt_pack_revision is None` instead of a dedicated
        # "already resolved" flag. `/start`'s truly-omitted-pack path
        # (the Cropwright default) resolves at request time to
        # `(active_name, None)` -- `None` there means "follow the active
        # pack's PINNED body dynamically" (`_get_vlm_labeler`/
        # `get_prompt_pack` redirect a bare active name with
        # `revision=None` to the pinned body, B1 round-2), NOT "not yet
        # resolved." Re-running `resolve_run_prompt_pack` on that
        # already-resolved bare `active_name` here re-entered the N7
        # "bare name -> latest saved revision" branch and silently pinned
        # an un-activated draft -- the round-1 B1 bug, reachable again via
        # the no-pack default path since the job no longer crashes.
        # `prompt_pack_resolved` is `/start`'s own explicit "I already ran
        # the resolver" signal, set unconditionally there -- so this
        # branch only fires for a call that never went through `/start`.
        #
        # R6-1b fix: capture "was `prompt_pack` omitted" from the RAW
        # incoming param before `resolve_run_prompt_pack` overwrites it
        # below -- this branch is the only place besides `/start` itself
        # that ever resolves an omitted pack, so it must compute the same
        # signal `/start` passes through explicitly.
        prompt_pack_omitted = not isinstance(prompt_pack, str)
        prompt_pack, prompt_pack_revision = await resolve_run_prompt_pack(opensearch, prompt_pack)
        if prompt_pack_omitted:
            # R7-2 fix: only the store-active case has TOCTOU draft risk.
            prompt_pack_omitted = omitted_pack_is_store_active(prompt_pack)
    # `/start` already resolved this `(prompt_pack, prompt_pack_revision)`
    # at request time when the branch above is skipped -- re-running
    # `resolve_run_prompt_pack` on the bare name here would silently
    # re-pin to whatever is CURRENTLY the latest saved revision (N7),
    # discarding the request's resolution (an explicit pin, or the
    # active-pack-follow `None` revision). Use it as-is.
    summary: dict[str, Any] = {
        'stages': {},
        'class_id': class_id,
        'cluster_id': cluster_id,
        'prompt_pack': prompt_pack,
        'prompt_pack_revision': prompt_pack_revision,
    }

    # Snapshot counts at entry for a real before/after.
    summary['baseline'] = await pipeline_health_snapshot(opensearch)

    scope_filter = ItemFilter(**(item_filter or {}))
    if embed_missing:
        # Lazy trigger: embed the in-scope items stored without a vector first.
        if progress is not None:
            progress.start_stage('embed_missing')
            progress.raise_if_cancelled()
        summary['stages']['embed_missing'] = await run_embed_missing_stage(
            opensearch,
            class_id=class_id,
            cluster_id=cluster_id,
            item_filter=item_filter,
            progress=progress,
        )

    # Pipeline order: classifier confident keeps its label; else -> VLM -> human.
    # force_cluster_id_equals_class_id keeps cluster_id==class_id for
    # labeled items; the residual clusterer handles the rest.
    # with_elapsed_tick advances the dashboard during callback-less
    # stages (update_by_query etc.).
    if train_clusters and cluster_id is not None:
        # Both stages rewrite cluster ids index-wide; a cluster-scoped job
        # writes only to the members it selected.
        for stage in ('cluster_id_normalize', 'cluster_residuals'):
            summary['stages'][stage] = dict(CLUSTER_SCOPED_SKIP)
    elif train_clusters:
        if progress is not None:
            progress.start_stage('cluster_id_normalize')
            progress.raise_if_cancelled()
        try:
            from src.services.curation.clustering.id_normalize import (
                force_cluster_id_equals_class_id,
            )

            normalize_result = await with_elapsed_tick(
                progress, force_cluster_id_equals_class_id(opensearch)
            )
            summary['stages']['cluster_id_normalize'] = normalize_result
        except Exception as exc:
            logger.warning('pipeline_cluster_normalize_failed', error=str(exc))
            summary['stages']['cluster_id_normalize'] = {
                'status': 'error',
                'error': str(exc),
            }

        # ---- stage 1b: AHC over residual pool ---------------------------
        if progress is not None:
            progress.start_stage('cluster_residuals')
            progress.raise_if_cancelled()
        # reassign_only: stream-assign residuals vs persisted IVF
        # centroids (no retrain). Else cluster_residuals dispatches via
        # the ClusterMethod registry (bad method -> stage error below).
        try:
            if reassign_only:
                from src.services.curation.clustering.orchestrator import assign_only_residuals

                cluster_result = await assign_only_residuals(opensearch, progress=progress)
            else:
                from src.services.curation.clustering.orchestrator import cluster_residuals

                cluster_result = await cluster_residuals(
                    opensearch,
                    recluster_unvalidated=recluster_unvalidated,
                    clustering_method=clustering_method,
                    gate_max_rank=gate_max_rank,
                    gate_min_blur_ratio=gate_min_blur_ratio,
                    n_clusters=n_clusters,
                    progress=progress,
                )
            summary['stages']['cluster_residuals'] = dict(cluster_result)
        except Exception as exc:
            logger.warning('pipeline_cluster_residuals_failed', error=str(exc))
            summary['stages']['cluster_residuals'] = {'status': 'error', 'error': str(exc)}
        from src.services.curation.clustering.cluster_geometry import cluster_geometry_stage

        # Every cluster (class ones too) gets centroid geometry.
        summary['stages']['cluster_residuals']['cluster_geometry'] = await with_elapsed_tick(
            progress, cluster_geometry_stage(opensearch)
        )

    # ---- stage 2: auto-promote ------------------------------------------
    if progress is not None:
        progress.start_stage('auto_promote')
        progress.raise_if_cancelled()
    if not run_auto_promote:
        # Disabled by default — the classifier+cluster-majority rule had
        # no classifier confidence floor and was contaminating class
        # clusters by validating low-confidence classifier predictions
        # as long as they sat in a cluster whose majority shared that
        # class. Skip the stage entirely until a confidence-gated
        # rewrite lands; operators can opt in via ?run_auto_promote=true.
        summary['stages']['auto_promote'] = {'skipped': True, 'reason': 'disabled by default'}
    elif cluster_id is not None:
        summary['stages']['auto_promote'] = dict(CLUSTER_SCOPED_SKIP)
    else:
        try:
            await opensearch.indices.refresh(index=items_index())
        except Exception as exc:
            logger.warning('pipeline_pre_promote_refresh_failed', error=str(exc))
        promote = await with_elapsed_tick(
            progress,
            gated_auto_promote(
                opensearch,
                min_purity=promote_min_purity,
                min_members=promote_min_members,
                dry_run=False,
            ),
        )
        summary['stages']['auto_promote'] = {
            'promoted': promote.get('promoted', 0),
            'skipped': promote.get('skipped', 0),
            'clusters_evaluated': len(promote.get('clusters', [])),
        }

    # ---- stage 3: VLM over remaining unvalidated ------------------------
    # Default-OFF: the detection worker's combined call already writes
    # ``class_id`` + ``vlm_verify_completed_at`` during region verification, so
    # a parallel VLM stage duplicates that work. Clusters stay numeric until the
    # worker labels their members and stage-1's
    # ``force_cluster_id_equals_class_id`` folds cluster_id -> class_id.
    # Opt-in: ``run_vlm=true`` backfills the no-region cohort (the segmenter
    # returned no candidate, so the combined call never fired).
    if not run_vlm:
        summary['stages']['vlm'] = skipped_vlm_stage()
        summary['stages']['cluster_id_normalize_post_vlm'] = {'skipped': True}
        if progress is not None:
            progress.start_stage('vlm', total=0)
            progress.start_stage('finalize')
        summary['unvalidated_remaining'] = await count_unvalidated_remaining(
            opensearch, items_index(), class_id, cluster_id, scope_filter
        )
        summary['after'] = await pipeline_health_snapshot(opensearch)
        return summary

    # Selection (sweep vs cluster scope, VLM policy, caps): autolabel/selection.py.
    guard = ClassWriteGuard('vlm_pipeline')
    try:
        selection = await resolve_vlm_selection(
            opensearch,
            class_id=class_id,
            cluster_id=cluster_id,
            classifier_confidence_skip_vlm=classifier_confidence_skip_vlm,
            item_filter=scope_filter,
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

    summary['stages']['unvalidated_after_promote'] = len(unvalidated_ids)

    if not unvalidated_ids:
        summary['stages']['vlm'] = skipped_vlm_stage(
            selection.empty_reason
            or await explain_empty_vlm_selection(
                opensearch, items_index(), class_id, cluster_id, scope_filter
            )
        )
        summary['final'] = {'unvalidated': 0, 'human_required': 0}
        return summary

    # Reuse the VLM label_batch logic by calling it directly (no HTTP
    # hop). Build ItemCrops here so we can chunk.
    from src.services.curation.region_class import item_classes
    from src.services.labeling.vlm_class_names import (
        format_class_catalog,
        resolve_class_name as _resolve_class_name_fn,
    )

    reg = get_class_registry().load()
    labelable = item_classes(reg.classes)
    class_names = [c.class_name for c in labelable]
    name_to_id = {c.class_name: c.class_id for c in labelable}

    # Render the registry as a grouped+described catalog so the VLM's
    # prompt tells it what each cryptic slug actually means visually. Big
    # quality lift over the bare CSV — see ``format_class_catalog``.
    class_dicts = [
        {'class_name': c.class_name, 'group': getattr(c, 'group', None)} for c in labelable
    ]
    # The run's selected pack (resolve_prompt_pack() default when unset).
    #
    # R6-1b fix: when the request omitted `prompt_pack`, resolve the
    # labeler against `None`/`None` -- ALWAYS "whatever is active right
    # now" -- never the echoed `(prompt_pack, prompt_pack_revision)`
    # name/revision, which can go stale between request time and here
    # (`prompt_pack`/`prompt_pack_revision` themselves are left untouched
    # for `summary`'s display).
    _labeler_pack, _labeler_revision = labeler_resolution_args(
        prompt_pack, prompt_pack_revision, prompt_pack_omitted=prompt_pack_omitted
    )
    endpoint = await job_endpoint(
        opensearch,
        vlm=vlm,
        acknowledge_external=acknowledge_external,
        pack=_resolve_pack(_labeler_pack, _labeler_revision),
        pinned_name=vlm_endpoint,
        pinned_revision=vlm_endpoint_revision,
        resolved=vlm_resolved,
    )
    if endpoint is not None:
        await assert_may_connect(endpoint)
    labeler = _get_vlm_labeler(_labeler_pack, _labeler_revision, endpoint=endpoint)
    class_catalog = format_class_catalog(class_dicts, labeler._pack)

    from src.services.labeling.vlm_prompts import prompt_pack_stamp

    _pack_stamp = prompt_pack_stamp(labeler._pack, revision=_labeler_revision)

    # Count how many crops bypass the synonym/fuzzy force-fit because the
    # VLM's confidence is low — those route straight to the raw-label
    # cluster pipeline instead. Logged in the pipeline summary.
    _force_fit_bypass = {'low_conf_skipped': 0, 'attempted': 0}

    def _resolve_class_name(raw: str, *, confidence: str | None = None) -> str | None:
        """Map a VLM reply to a registry class name (or None).

        Skips fuzzy/synonym resolution when ``confidence='low'`` — see
        :func:`src.services.labeling.vlm_class_names.resolve_class_name`.
        """
        if confidence == 'low':
            _force_fit_bypass['low_conf_skipped'] += 1
        else:
            _force_fit_bypass['attempted'] += 1
        return _resolve_class_name_fn(raw, name_to_id, confidence=confidence)  # type: ignore[arg-type]

    # Prototype-rescue paths are deleted: CLIP-prototype labeling
    # mis-labeled a large fraction of rows in an earlier phase. The
    # classifier+VLM agreement two-signal path (`class_source='classifier_vlm_agreement'`)
    # is a documented follow-up. For this slice, the VLM writes
    # `class_source='vlm'` (or `vlm_unmatched` / `vlm_new_class_pending`)
    # WITHOUT auto-validation. Validation requires either a human signal
    # or the classifier+VLM two-signal path.

    from src.clients.occ import occ_skip_on_conflict_bulk as _occ_skip_bulk
    from src.services.curation.vlm_class_attempt import prediction_class_update, with_class_snapshot

    async def _run_chunk(ids: list[str]) -> tuple[int, int, list[dict[str, Any]]]:
        # A single mget_crops call per chunk instead of per-crop
        # opensearch.get — far fewer round-trips.
        docs = await mget_crops(
            opensearch,
            ids,
            source_includes=['image_path', 'bbox_norm'],
        )
        crops: list[ItemCrop] = []
        for crop_id, doc in docs.items():
            src = doc.get('_source') or {}
            image_path = src.get('image_path', '')
            bbox = src.get('bbox_norm')
            if not image_path or not bbox or len(bbox) != 4:
                continue
            try:
                from src.services.curation.image_serving import resolve_crop_root, resolve_safe_path

                safe = resolve_safe_path(image_path, resolve_crop_root(image_path))
                jpeg = THUMBNAIL_CACHE.get_or_compute(safe, tuple(bbox), size=224)
            except Exception as exc:
                logger.warning('pipeline_thumb_failed', crop_id=crop_id, error=str(exc))
                continue
            crops.append(ItemCrop(img_id=crop_id, jpeg_bytes=jpeg))
        # No item classes yet (a fresh project, or only the region class):
        # nothing to label items as.
        if not crops or not class_names:
            return 0, 0, []
        preds = await labeler.label_or_propose_batch(
            crops,
            class_names,
            class_catalog=class_catalog,
        )
        # Collect per-doc updates, then dispatch via
        # occ_skip_on_conflict_bulk so a concurrent human edit always
        # wins. ``updates_by_id`` keys the merger by crop_id so the OCC
        # helper can rebuild the doc against the fresh _source.
        updates_by_id: dict[str, dict[str, Any]] = {}
        proposals: list[dict[str, Any]] = []
        now = _now_iso()
        from src.services.detection.cascade_detect import class_provenance as _class_prov

        _vlm_class_prov = _class_prov(
            detector=labeler.identity.model,
            detector_version='1',
            labeler=labeler.identity.model,
            labeled_at=now,
        )
        for p in preds:
            # Always write the VLM's make/model/region_visible when
            # reported, regardless of which class-resolution path fires.
            _vlm_extras: dict[str, Any] = {}
            if p.make:
                _vlm_extras['vlm_item_make'] = p.make
            if p.model:
                _vlm_extras['vlm_item_model'] = p.model
            if p.region_visible is not None:
                _vlm_extras[get_region_fields().visible] = p.region_visible
            update, proposal = prediction_class_update(
                p,
                name_to_id=name_to_id,
                resolve=_resolve_class_name,
                now=now,
                provenance=_vlm_class_prov,
                extras=_vlm_extras,
                set_cluster=True,
            )
            if update is None:
                continue
            update['vlm_prompt_pack'] = _pack_stamp
            update.update(_vlm_stamp(labeler))
            if proposal is not None:
                proposals.append(proposal)
            updates_by_id[p.img_id] = update
        if updates_by_id:
            # Worker-context bulk write via OCC. Human edits always win
            # on conflict; class_id_history snapshots the prior
            # assignment when this write changes class_id.
            def _merge_pipeline(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
                # Only onto the class state the sweep selected on: a human
                # write (or validation) since the scroll wins.
                if not guard.allows(doc_id, current):
                    return {}
                return with_class_snapshot(
                    dict(updates_by_id[doc_id]), current, writer='vlm_pipeline'
                )

            try:
                await _occ_skip_bulk(
                    opensearch,
                    doc_ids=list(updates_by_id.keys()),
                    merger=_merge_pipeline,
                    index=items_index(),
                    refresh=False,
                    writer_id='vlm_pipeline',
                )
            except Exception as exc:
                logger.warning('pipeline_bulk_failed', error=str(exc))
            else:
                # Live UI updates per written crop.
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
        return len(preds), len(updates_by_id), proposals

    chunks = [
        unvalidated_ids[i : i + vlm_batch_size]
        for i in range(0, len(unvalidated_ids), vlm_batch_size)
    ]
    sem = _asyncio.Semaphore(vlm_concurrency)
    # The VLM is the dominant cost at scale (often hours). Report
    # per-chunk progress so the labeler's progress bar moves visibly, and
    # check for operator cancel between chunks so a cancel takes effect
    # within a few seconds rather than waiting for the whole gather to
    # finish.
    if progress is not None:
        progress.start_stage('vlm', total=len(unvalidated_ids))

    async def _gated(ids: list[str]) -> tuple[int, int, list[dict[str, Any]]]:
        async with sem:
            if progress is not None:
                progress.raise_if_cancelled()
            out = await _run_chunk(ids)
            if progress is not None:
                progress.advance(len(ids))
            return out

    results = await _asyncio.gather(*[_gated(c) for c in chunks])
    g_predicted = sum(r[0] for r in results)
    g_updated = sum(r[1] for r in results)
    new_class_proposals: list[dict[str, Any]] = []
    for _, _, props in results:
        new_class_proposals.extend(props)

    # Force a refresh so subsequent reads see the updates.
    try:
        await opensearch.indices.refresh(index=items_index())
    except Exception as exc:
        logger.warning('pipeline_refresh_failed', error=str(exc))

    # Final cluster_id normalization — the VLM may have changed class_id
    # on items without rewriting cluster_id, which fragments the labeler
    # view. One last pass ensures cluster_id equals class_id everywhere a
    # class is set — only on the selected items when the job is scoped.
    if progress is not None:
        progress.start_stage('finalize')
    try:
        from src.services.curation.clustering.id_normalize import (
            force_cluster_id_equals_class_id as _force_cluster_eq_class,
        )

        scoped = cluster_id is not None or class_id is not None
        post_normalize = await with_elapsed_tick(
            progress,
            _force_cluster_eq_class(opensearch, crop_ids=unvalidated_ids if scoped else None),
        )
        summary['stages']['cluster_id_normalize_post_vlm'] = post_normalize
    except Exception as exc:
        logger.warning('pipeline_post_normalize_failed', error=str(exc))

    # Re-count what's still unvalidated for the dashboard.
    try:
        cnt_resp = await opensearch.count(
            index=items_index(),
            body={
                'query': unvalidated_count_query(
                    class_id=class_id, cluster_id=cluster_id, item_filter=scope_filter
                )
            },
        )
        remaining = int(cnt_resp.get('count', 0))
    except Exception:
        remaining = -1

    # Surface how often we skipped the synonym/fuzzy force-fit.
    # ``low_conf_skipped`` counts items where the VLM replied with low
    # confidence and we declined to spend cycles trying to map the answer
    # onto a registry slot — those items fall through to
    # ``vlm_unmatched`` with the raw label preserved for downstream
    # clustering.
    logger.info(
        'pipeline_force_fit_bypass',
        attempted=_force_fit_bypass['attempted'],
        low_conf_skipped=_force_fit_bypass['low_conf_skipped'],
    )
    summary['stages']['vlm'] = {
        'predicted': g_predicted,
        'updated': g_updated,
        'new_class_proposals': new_class_proposals[:50],
        'new_class_proposals_total': len(new_class_proposals),
        'force_fit_bypass': dict(_force_fit_bypass),
    }
    summary['final'] = {
        'unvalidated': remaining,
        'human_required': remaining,
    }
    # Capture the after-state and compute deltas the labeler can render
    # as "before/after" without re-querying.
    summary['after'] = await pipeline_health_snapshot(opensearch)
    baseline = summary.get('baseline') or {}
    after = summary['after']
    summary['delta'] = {
        k: int(after.get(k, 0)) - int(baseline.get(k, 0))
        for k in ('validated', 'unvalidated', 'has_class', 'cluster_id_mismatched')
        if k in after and k in baseline
    }
    return summary


# POST /pipeline/auto_label (the public route) lives in pipeline_public.py
# (700-LOC ratchet split, same pattern as _region_profile_clone.py).
# Re-exported as `pipeline.pipeline_auto_label` too (not just imported for
# route-registration side effects) -- many existing tests call this name
# directly as `pipeline.pipeline_auto_label(...)`, predating the split.
from src.routers.curation.pipeline_public import pipeline_auto_label  # noqa: E402,F401
