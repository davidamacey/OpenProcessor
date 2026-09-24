"""Curation auto-label router (SSE: pipeline_events; status/cancel: pipeline_control)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Query

from src.clients.curation_opensearch import mget_crops
from src.config import get_region_fields
from src.config.curation import ITEM_EMBEDDING_FIELD
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    _now_iso,
    get_class_registry,
    logger,
    router,
)
from src.routers.curation.pipeline_params import (
    AUTO_PROMOTE_DESC as _AUTO_PROMOTE_DESC,
    CLASS_ID_DESC as _CLASS_ID_DESC,
    CLUSTER_ID_DESC as _CLUSTER_ID_DESC,
    PROMPT_PACK_DESC as _PROMPT_PACK_DESC,
    REASSIGN_ONLY_DESC as _REASSIGN_ONLY_DESC,
    RUN_VLM_DESC as _RUN_VLM_DESC,
    reject_detection_profile,
    resolve_run_prompt_pack,
)
from src.routers.curation.vlm import _get_vlm_labeler
from src.services.curation.autolabel.selection import unvalidated_count_query, vlm_selection_query
from src.services.curation.class_write_guard import CLASS_GUARD_SOURCE_FIELDS, ClassWriteGuard
from src.services.curation.cluster_purity import PROMOTE_MIN_MEMBERS, PROMOTE_MIN_PURITY
from src.services.curation.event_hub import publish_crop_classified


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


@router.post('/pipeline/auto_label/start')
async def pipeline_auto_label_start(
    opensearch: OpenSearchDep,
    train_clusters: Annotated[bool, Query()] = True,
    promote_min_purity: Annotated[float, Query(ge=0.5, le=1.0)] = PROMOTE_MIN_PURITY,
    promote_min_members: Annotated[int, Query(ge=2, le=1000)] = PROMOTE_MIN_MEMBERS,
    vlm_batch_size: Annotated[int, Query(ge=4, le=64)] = 32,
    vlm_concurrency: Annotated[int, Query(ge=1, le=128)] = 16,
    max_vlm_crops: Annotated[int, Query(ge=0, le=100000)] = 0,
    classifier_confidence_skip_vlm: Annotated[float, Query(ge=0.0, le=1.0)] = 0.80,
    clustering_method: Annotated[str | None, Query()] = None,
    run_vlm: Annotated[bool, Query(description=_RUN_VLM_DESC)] = False,
    recluster_unvalidated: Annotated[bool, Query(description='Merge candidate clusters.')] = False,
    run_auto_promote: Annotated[bool, Query(description=_AUTO_PROMOTE_DESC)] = False,
    reassign_only: Annotated[bool, Query(description=_REASSIGN_ONLY_DESC)] = False,
    # Cluster scope (primary-subject gate).
    gate_max_rank: Annotated[int | None, Query(ge=1)] = None,
    gate_min_blur_ratio: Annotated[float | None, Query(ge=0.0)] = None,
    n_clusters: Annotated[int | None, Query(ge=2, le=4096)] = None,
    class_id: Annotated[int | None, Query(description=_CLASS_ID_DESC)] = None,
    cluster_id: Annotated[int | None, Query(description=_CLUSTER_ID_DESC)] = None,
    detection_profile: Annotated[str | None, Query(include_in_schema=False)] = None,
    prompt_pack: Annotated[str | None, Query(description=_PROMPT_PACK_DESC)] = None,
) -> dict[str, Any]:
    """Kick off auto_label as a background job. Returns immediately.

    The labeler polls ``GET /pipeline/auto_label/status`` for progress.
    Only one job runs at a time; a second start request returns HTTP 409
    while a job is in flight.
    """
    from src.services.curation.autolabel import job as auto_label_job

    # Resolved here (422 before queueing) so the job args echo what runs.
    reject_detection_profile(detection_profile)
    prompt_pack = await resolve_run_prompt_pack(opensearch, prompt_pack)
    try:
        return auto_label_job.start_job(
            pipeline_auto_label,
            {
                'opensearch': opensearch,
                'train_clusters': train_clusters,
                'promote_min_purity': promote_min_purity,
                'promote_min_members': promote_min_members,
                'vlm_batch_size': vlm_batch_size,
                'vlm_concurrency': vlm_concurrency,
                'max_vlm_crops': max_vlm_crops,
                'classifier_confidence_skip_vlm': classifier_confidence_skip_vlm,
                'clustering_method': clustering_method,
                'run_vlm': run_vlm,
                'recluster_unvalidated': recluster_unvalidated,
                'run_auto_promote': run_auto_promote,
                'reassign_only': reassign_only,
                'gate_max_rank': gate_max_rank,
                'gate_min_blur_ratio': gate_min_blur_ratio,
                'n_clusters': n_clusters,
                'class_id': class_id,
                'cluster_id': cluster_id,
                'prompt_pack': prompt_pack,
            },
        )
    except RuntimeError as exc:
        # 409 makes it unambiguous in the UI that a run is already in flight.
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post('/pipeline/auto_label')
async def pipeline_auto_label(
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
    """
    import asyncio as _asyncio

    from src.routers.curation.pipeline_health import pipeline_health_snapshot
    from src.services.curation.autolabel.job import with_elapsed_tick
    from src.services.curation.clustering.auto_promote import auto_promote_clusters
    from src.services.curation.image_serving import THUMBNAIL_CACHE
    from src.services.labeling.vlm_labeler import ItemCrop

    reject_detection_profile(detection_profile)
    prompt_pack = await resolve_run_prompt_pack(opensearch, prompt_pack)
    summary: dict[str, Any] = {
        'stages': {},
        'class_id': class_id,
        'cluster_id': cluster_id,
        'prompt_pack': prompt_pack,
    }

    # Snapshot counts at entry for a real before/after.
    summary['baseline'] = await pipeline_health_snapshot(opensearch)

    # Pipeline order: v6 confident keeps its label; else -> VLM -> human.
    # force_cluster_id_equals_class_id keeps cluster_id==class_id for
    # labeled items; the residual clusterer handles the rest.
    # with_elapsed_tick advances the dashboard during callback-less
    # stages (update_by_query etc.).
    if train_clusters:
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

    # ---- stage 2: auto-promote ------------------------------------------
    if progress is not None:
        progress.start_stage('auto_promote')
        progress.raise_if_cancelled()
    if not run_auto_promote:
        # Disabled by default — the v6+cluster-majority rule had no v6
        # confidence floor and was contaminating class clusters by
        # validating low-confidence v6 predictions as long as they sat in
        # a cluster whose majority shared that class. Skip the stage
        # entirely until a confidence-gated rewrite lands; operators can
        # opt in via ?run_auto_promote=true.
        summary['stages']['auto_promote'] = {'skipped': True, 'reason': 'disabled by default'}
    else:
        try:
            await opensearch.indices.refresh(index=CURATION_ITEMS_INDEX)
        except Exception as exc:
            logger.warning('pipeline_pre_promote_refresh_failed', error=str(exc))
        promote = await with_elapsed_tick(
            progress,
            auto_promote_clusters(
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
    # Default-OFF: the detection worker's combined call writes ``class_id``
    # + ``vlm_verify_completed_at`` as a side effect of region
    # verification, so a parallel auto_label VLM stage just duplicates
    # work the worker is already doing on every drain pass. Clusters here
    # remain numeric until the worker eventually labels their members and
    # stage-1's ``force_cluster_id_equals_class_id`` folds cluster_id ->
    # class_id.
    #
    # Opt-in: pass ``run_vlm=true`` for a one-off backfill of the
    # no-region cohort (items where the segmenter returned no candidate,
    # so the combined call never fired and no class label was written).
    if not run_vlm:
        summary['stages']['vlm'] = {'skipped': True, 'predicted': 0, 'updated': 0}
        summary['stages']['cluster_id_normalize_post_vlm'] = {'skipped': True}
        if progress is not None:
            progress.start_stage('vlm', total=0)
            progress.start_stage('finalize')
        try:
            body = {'query': unvalidated_count_query(class_id=class_id, cluster_id=cluster_id)}
            cnt = await opensearch.count(index=CURATION_ITEMS_INDEX, body=body)
            summary['unvalidated_remaining'] = int(cnt.get('count', 0))
        except Exception:
            summary['unvalidated_remaining'] = -1
        summary['after'] = await pipeline_health_snapshot(opensearch)
        return summary

    # Selection (global sweep vs cluster scope) lives in
    # services/curation/autolabel/selection.py. Scroll the FULL scope:
    # max_vlm_crops == 0 processes everything, > 0 caps the run.
    SCROLL_PAGE = 1000
    SCROLL_TTL = '5m'
    unvalidated_query = vlm_selection_query(
        class_id=class_id,
        cluster_id=cluster_id,
        classifier_confidence_skip_vlm=classifier_confidence_skip_vlm,
    )
    initial_body = {
        'size': SCROLL_PAGE,
        '_source': list(VLM_SWEEP_SOURCE_FIELDS),
        'query': unvalidated_query,
        'sort': [{'updated_at': 'asc'}],
    }
    cap = max_vlm_crops if max_vlm_crops > 0 else None
    unvalidated_ids: list[str] = []
    guard = ClassWriteGuard('vlm_pipeline')
    scroll_id: str | None = None
    try:
        resp = await opensearch.search(
            index=CURATION_ITEMS_INDEX, body=initial_body, scroll=SCROLL_TTL
        )
        while True:
            scroll_id = resp.get('_scroll_id')
            hits = (resp.get('hits') or {}).get('hits') or []
            if not hits:
                break
            for h in hits:
                cid = (h.get('_source') or {}).get('crop_id') or h.get('_id')
                if cid:
                    unvalidated_ids.append(cid)
                    guard.remember(cid, h.get('_source') or {})
                    if cap is not None and len(unvalidated_ids) >= cap:
                        break
            if cap is not None and len(unvalidated_ids) >= cap:
                break
            if not scroll_id:
                break
            resp = await opensearch.scroll(scroll_id=scroll_id, scroll=SCROLL_TTL)
    except Exception as exc:
        return {**summary, 'stages_error': f'fetch unvalidated failed: {exc}'}
    finally:
        if scroll_id:
            try:
                await opensearch.clear_scroll(scroll_id=scroll_id)
            except Exception as exc:
                logger.debug('pipeline_clear_scroll_failed', error=str(exc))

    summary['stages']['unvalidated_after_promote'] = len(unvalidated_ids)

    if not unvalidated_ids:
        summary['stages']['vlm'] = {'predicted': 0, 'updated': 0}
        summary['final'] = {'unvalidated': 0, 'human_required': 0}
        return summary

    # Reuse the VLM label_batch logic by calling it directly (no HTTP
    # hop). Build ItemCrops here so we can chunk.
    from src.services.labeling.vlm_labeler import (
        format_class_catalog,
        resolve_class_name as _resolve_class_name_fn,
    )

    reg = get_class_registry().load()
    class_names = [c.class_name for c in reg.classes if not c.deprecated]
    name_to_id = {c.class_name: c.class_id for c in reg.classes if not c.deprecated}

    # Render the registry as a grouped+described catalog so the VLM's
    # prompt tells it what each cryptic slug actually means visually. Big
    # quality lift over the bare CSV — see ``format_class_catalog``.
    class_dicts = [
        {'class_name': c.class_name, 'group': getattr(c, 'group', None)}
        for c in reg.classes
        if not c.deprecated
    ]
    # The run's selected pack (resolve_prompt_pack() default when unset).
    labeler = _get_vlm_labeler(prompt_pack)
    class_catalog = format_class_catalog(class_dicts, labeler._pack)

    # Count how many crops bypass the synonym/fuzzy force-fit because the
    # VLM's confidence is low — those route straight to the raw-label
    # cluster pipeline instead. Logged in the pipeline summary.
    _force_fit_bypass = {'low_conf_skipped': 0, 'attempted': 0}

    def _resolve_class_name(raw: str, *, confidence: str | None = None) -> str | None:
        """Map a VLM reply to a registry class name (or None).

        Skips fuzzy/synonym resolution when ``confidence='low'`` — see
        :func:`src.services.labeling.vlm_labeler.resolve_class_name`.
        """
        if confidence == 'low':
            _force_fit_bypass['low_conf_skipped'] += 1
        else:
            _force_fit_bypass['attempted'] += 1
        return _resolve_class_name_fn(raw, name_to_id, confidence=confidence)  # type: ignore[arg-type]

    # Prototype-rescue paths are deleted: CLIP-prototype labeling
    # mis-labeled a large fraction of rows in an earlier phase. The
    # v6+VLM agreement two-signal path (`class_source='classifier_vlm_agreement'`)
    # is a documented follow-up. For this slice, the VLM writes
    # `class_source='vlm'` (or `vlm_unmatched` / `vlm_new_class_pending`)
    # WITHOUT auto-validation. Validation requires either a human signal
    # or the v6+VLM two-signal path.

    from src.clients.occ import occ_skip_on_conflict_bulk as _occ_skip_bulk
    from src.services.curation.history import record_class_history as _record_history

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
        if not crops:
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
            detector=labeler.model,
            detector_version='1',
            labeler=labeler.model,
            labeled_at=now,
        )
        for p in preds:
            # Capture the VLM's raw answer on EVERY prediction so we can
            # later aggregate the long tail and grow the registry. Even
            # when the response was unparseable we keep p.raw_response
            # (best-effort excerpt) so v6-missed open-vocabulary
            # classifications still seed the next training round. The
            # ``__new__`` sentinel itself is not informative — prefer the
            # populated ``proposed_class`` slug when present, then the
            # parsed class_name, finally the raw model output.
            raw_label = (
                p.proposed_class if p.class_name == '__new__' else (p.class_name or p.raw_response)
            )

            # Always write the VLM's make/model/region_visible when
            # reported, regardless of which class-resolution path fires.
            _vlm_extras: dict[str, Any] = {}
            if p.make:
                _vlm_extras['vlm_item_make'] = p.make
            if p.model:
                _vlm_extras['vlm_item_model'] = p.model
            if p.plate_visible is not None:
                _vlm_extras[get_region_fields().visible] = p.plate_visible

            if p.class_name == '__new__' and p.proposed_class:
                proposed_resolved = _resolve_class_name(p.proposed_class, confidence=p.confidence)
                if proposed_resolved is not None:
                    # The VLM's slug resolved via synonyms — normal prediction.
                    cid = name_to_id[proposed_resolved]
                    updates_by_id[p.img_id] = {
                        'class_id': cid,
                        'class_name': proposed_resolved,
                        'class_source': 'vlm',
                        # Clear stale label_source so a prior auto_promote
                        # validation tag can't survive the VLM overwrite.
                        'label_source': 'vlm',
                        'vlm_confidence': p.confidence,
                        'vlm_raw_class': p.proposed_class,
                        'vlm_raw_label': raw_label,
                        **_vlm_extras,
                        **_vlm_class_prov,
                        'updated_at': now,
                    }
                    continue
                # Truly new — surface for the curator queue.
                proposals.append({'crop_id': p.img_id, 'proposed_class': p.proposed_class})
                updates_by_id[p.img_id] = {
                    'class_source': 'vlm_new_class_pending',
                    'label_source': 'vlm',
                    'vlm_proposed_class': p.proposed_class,
                    'vlm_raw_label': raw_label,
                    'vlm_confidence': p.confidence,
                    'needs_new_class': True,
                    **_vlm_extras,
                    'updated_at': now,
                }
                continue
            resolved = _resolve_class_name(p.class_name, confidence=p.confidence)
            if resolved is None:
                updates_by_id[p.img_id] = {
                    'class_source': 'vlm_unmatched',
                    'label_source': 'vlm',
                    'vlm_raw_class': p.class_name,
                    'vlm_raw_label': raw_label,
                    'vlm_confidence': p.confidence,
                    **_vlm_extras,
                    'updated_at': now,
                }
                continue
            cid = name_to_id[resolved]
            updates_by_id[p.img_id] = {
                'class_id': cid,
                'class_name': resolved,
                'class_source': 'vlm',
                'label_source': 'vlm',
                'vlm_confidence': p.confidence,
                'vlm_raw_label': raw_label,
                'cluster_id': cid,
                **_vlm_extras,
                **_vlm_class_prov,
                'updated_at': now,
            }
        if updates_by_id:
            # Worker-context bulk write via OCC. Human edits always win
            # on conflict; class_id_history snapshots the prior
            # assignment when this write changes class_id.
            def _merge_pipeline(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
                # Only onto the class state the sweep selected on: a human
                # write (or validation) since the scroll wins.
                if not guard.allows(doc_id, current):
                    return {}
                update = dict(updates_by_id[doc_id])
                if 'class_id' in update:
                    update['class_id_history'] = _record_history(current, writer='vlm_pipeline')
                return update

            try:
                await _occ_skip_bulk(
                    opensearch,
                    doc_ids=list(updates_by_id.keys()),
                    merger=_merge_pipeline,
                    index=CURATION_ITEMS_INDEX,
                    refresh=False,
                    writer_id='vlm_pipeline',
                )
            except Exception as exc:
                logger.warning('pipeline_bulk_failed', error=str(exc))
            else:
                # Live UI updates per written crop.
                for raw_id, doc in updates_by_id.items():
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
        await opensearch.indices.refresh(index=CURATION_ITEMS_INDEX)
    except Exception as exc:
        logger.warning('pipeline_refresh_failed', error=str(exc))

    # Final cluster_id normalization — the VLM may have changed class_id
    # on items without rewriting cluster_id, which fragments the labeler
    # view. One last pass ensures cluster_id equals class_id everywhere a
    # class is set.
    if progress is not None:
        progress.start_stage('finalize')
    try:
        from src.services.curation.clustering.id_normalize import (
            force_cluster_id_equals_class_id as _force_cluster_eq_class,
        )

        post_normalize = await with_elapsed_tick(progress, _force_cluster_eq_class(opensearch))
        summary['stages']['cluster_id_normalize_post_vlm'] = post_normalize
    except Exception as exc:
        logger.warning('pipeline_post_normalize_failed', error=str(exc))

    # Re-count what's still unvalidated for the dashboard.
    try:
        cnt_resp = await opensearch.count(
            index=CURATION_ITEMS_INDEX,
            body={'query': unvalidated_count_query(class_id=class_id, cluster_id=cluster_id)},
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
