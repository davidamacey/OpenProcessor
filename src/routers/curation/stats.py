"""Dataset stats endpoints for the curation labeler dashboard.

Split out of :mod:`src.routers.curation.review` to keep individual
router files under the 700 LOC pre-commit ceiling. Two endpoints:

* ``GET /curation/stats/classes`` — per-class total/validated breakdown.
* ``GET /curation/stats/dataset`` — overall dataset balance + label-source +
  pipeline breakdown. Single ``_search?size=0`` powering the labeler's
  pipeline dashboard.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.config.region_fields import RegionFields, get_region_fields
from src.config.region_state import RegionStatus
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    _now_iso,
    get_class_registry,
    router,
)
from src.services.curation.dataset_thresholds import adequacy, aug_target, dataset_thresholds
from src.services.curation.ingest_class_sources import (
    CLASSIFIER_VLM_AGREEMENT_CLASS_SOURCE,
    CLUSTER_MAJORITY_CLASS_SOURCE,
    DEFAULT_PROPOSAL_CLASS_SOURCE,
    VLM_CLASS_SOURCE,
    classifier_class_sources,
    unlabeled_proposal_class_sources,
)
from src.services.detection.profile_registry import region_profile_or_neutral


@router.get('/stats/classes')
async def stats_classes(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Per-class total/validated breakdown.

    Returns both the raw ``by_class`` aggregation and a flattened ``classes``
    array (``{class_id, class_name, count, validated_count, trainable,
    trainable_gap, adequacy, aug_target, aug_gap}``, plus the ``thresholds``
    behind them) joined against the registry for the names. The labeler's
    ``getStats()`` (Export + Home pages) consumes ``classes`` — the
    aggregation alone has no class names, so without this join the Export
    dataset table renders empty.

    ``trainable`` = validated minus frozen test-holdout minus human-excluded
    crops — the frontend used to compute ``validated - holdout`` itself
    client-side; this serves the real number (also excluding
    ``class_excluded`` crops, which the client-side version didn't
    account for) so every page agrees. ``trainable_gap`` is the shortfall
    against :func:`dataset_thresholds`'s per-class minimum, floored at 0.
    """
    body = {
        'size': 0,
        'aggs': {
            'by_class': {
                'terms': {'field': 'class_id', 'size': 1000},
                'aggs': {
                    'validated': {'filter': {'term': {'class_validated': True}}},
                    # label_source is mapped keyword directly on the live
                    # index — no .keyword subfield exists.
                    'by_source': {'terms': {'field': 'label_source', 'size': 16}},
                    # Trainable = validated minus these two: a frozen
                    # test-holdout crop must never leak into training, and
                    # a human-excluded crop was deliberately removed from
                    # the dataset.
                    'validated_test_holdout': {
                        'filter': {
                            'bool': {
                                'filter': [
                                    {'term': {'class_validated': True}},
                                    {'term': {'test_holdout': True}},
                                ]
                            }
                        }
                    },
                    'validated_excluded': {
                        'filter': {
                            'bool': {
                                'filter': [
                                    {'term': {'class_validated': True}},
                                    {'term': {'class_excluded': True}},
                                ]
                            }
                        }
                    },
                },
            }
        },
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc
    aggs = resp.get('aggregations') or {}

    counts: dict[int, int] = {}
    validated: dict[int, int] = {}
    validated_test_holdout: dict[int, int] = {}
    validated_excluded: dict[int, int] = {}
    for bucket in aggs.get('by_class', {}).get('buckets', []):
        cid = int(bucket['key'])
        counts[cid] = int(bucket.get('doc_count', 0))
        validated[cid] = int((bucket.get('validated') or {}).get('doc_count', 0))
        validated_test_holdout[cid] = int(
            (bucket.get('validated_test_holdout') or {}).get('doc_count', 0)
        )
        validated_excluded[cid] = int((bucket.get('validated_excluded') or {}).get('doc_count', 0))

    threshold_values = dataset_thresholds()
    hard_min = int(threshold_values.get('block_below', 0))

    classes: list[dict[str, Any]] = []
    try:
        reg = get_class_registry().load()
        for c in reg.classes:
            if getattr(c, 'deprecated', False):
                continue
            n_valid = validated.get(c.class_id, 0)
            n_trainable = (
                n_valid
                - validated_test_holdout.get(c.class_id, 0)
                - validated_excluded.get(c.class_id, 0)
            )
            target = aug_target(n_valid)
            classes.append(
                {
                    'class_id': c.class_id,
                    'class_name': c.class_name,
                    'count': counts.get(c.class_id, 0),
                    'validated_count': n_valid,
                    'trainable': n_trainable,
                    'trainable_gap': max(0, hard_min - n_trainable),
                    'adequacy': adequacy(n_valid),
                    'aug_target': target,
                    'aug_gap': target - n_valid,
                }
            )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'class registry unavailable: {exc}') from exc

    return {**aggs, 'classes': classes, 'thresholds': threshold_values}


# Provenance keys that mean "human-applied or human-validated", used to
# avoid double-counting class_source buckets against the ``by_human``
# rollup which is already computed from class_validated / the region
# validated flag.
_HUMAN_SOURCE_PREFIXES = ('human',)


def _sum_prefixed(buckets: dict[str, int], prefix: str) -> int:
    """Sum bucket counts whose key starts with ``prefix``. An empty prefix
    (e.g. no detector configured) matches nothing rather than everything."""
    if not prefix:
        return 0
    return sum(cnt for k, cnt in buckets.items() if k.startswith(prefix))


def _rollup_class_sources(buckets: list[dict[str, Any]]) -> dict[str, int]:
    """Roll up ``class_source`` buckets into dashboard taxonomy.

    Returns a dict of class label provenance counts. ``by_human``
    counts ONLY rows whose ``class_source`` is a human-prefixed value
    (e.g. 'human', 'human_move'). A majority-agreement auto-validator
    also sets ``class_validated=True`` but that's auto-validation, not a
    human label — using the validated flag here would inflate by_human.
    Region-detector breakdown (detector / segmenter / human) is computed
    separately by the caller via region-detector aggregations.

    Callers pass ONLY ``class_sources_with_class`` buckets (docs that
    carry a class_id — see ``_build_dataset_query_body``). F-23: this
    function used to also bucket a ``by_proposal`` count here, but the
    unlabeled-proposal class_source values (``unlabeled_proposal_class_sources``)
    are, by construction, the detector's "no class assigned yet" marker
    -- such a doc never carries a class_id, so it can never appear in
    ``class_sources_with_class``. ``by_proposal`` was therefore always 0
    on every real deployment. See ``_count_by_proposal`` for the fixed
    accounting, now sourced from the class-less buckets and reported
    under ``unlabeled`` instead.
    """
    classifier_sources = classifier_class_sources() | {
        CLUSTER_MAJORITY_CLASS_SOURCE,
        CLASSIFIER_VLM_AGREEMENT_CLASS_SOURCE,
    }
    by_human = 0
    by_vlm = 0
    by_classifier = 0
    by_other = 0
    for b in buckets:
        key = str(b.get('key', ''))
        cnt = int(b.get('doc_count', 0))
        if key.startswith(_HUMAN_SOURCE_PREFIXES):
            by_human += cnt
        elif key.startswith(VLM_CLASS_SOURCE):
            by_vlm += cnt
        elif key in classifier_sources:
            by_classifier += cnt
        else:
            # Truly unknown / future provenance — surface in 'other'
            # rather than dropping.
            by_other += cnt
    return {
        'by_human': by_human,
        'by_vlm': by_vlm,
        'by_classifier': by_classifier,
        'other': by_other,
    }


def _count_by_proposal(buckets: list[dict[str, Any]]) -> int:
    """Sum ``class_source`` buckets that are a detector's "proposed but not
    yet classified" marker (``unlabeled_proposal_class_sources()`` plus the
    ingest-model default). Takes ``class_sources_no_class`` buckets (docs
    with NO class_id) -- these sources never carry a class_id, so counting
    them here (not in ``_rollup_class_sources``) is what makes the number
    non-zero. A subset of ``unlabeled.no_label_source``, surfaced
    explicitly the same way ``vlm_no_class`` is."""
    proposal_sources = unlabeled_proposal_class_sources() | {DEFAULT_PROPOSAL_CLASS_SOURCE}
    return sum(
        int(b.get('doc_count', 0)) for b in buckets if str(b.get('key', '')) in proposal_sources
    )


def _read_auto_label_clusters_meta(
    fallback_residual: int,
    fallback_noise: int,
) -> dict[str, Any]:
    """Best-effort read of the auto_label_job persisted state file.

    Returns a dict with ``last_run_at``, ``cluster_count`` (the run's own
    count; the caller replaces it with the index total and moves this to
    ``last_run_cluster_count``), ``residual_count``, ``noise_count``,
    ``method``. Any read / parse
    failure falls back to (None, 0, fallback_residual, fallback_noise,
    None) — the stats endpoint must never 500 because the on-disk
    state file is missing or malformed.
    """
    last_run_at: str | None = None
    cluster_count = 0
    residual_count = fallback_residual
    noise_count = fallback_noise
    method: str | None = None
    try:
        from src.services.curation.autolabel.job import get_state as _get_state
    except ImportError:
        # auto_label_job module not importable — surface no meta.
        return {
            'last_run_at': last_run_at,
            'cluster_count': cluster_count,
            'residual_count': residual_count,
            'noise_count': noise_count,
            'method': method,
        }
    try:
        st = _get_state() or {}
    except Exception:
        # Disk read / JSON parse error — leave defaults.
        return {
            'last_run_at': last_run_at,
            'cluster_count': cluster_count,
            'residual_count': residual_count,
            'noise_count': noise_count,
            'method': method,
        }

    finished_at = st.get('finished_at') or 0
    if finished_at:
        try:
            from datetime import UTC, datetime

            last_run_at = datetime.fromtimestamp(float(finished_at), UTC).isoformat()
        except (TypeError, ValueError, OSError):
            last_run_at = None

    stages = ((st.get('result') or {}).get('stages')) or {}
    # ``cluster_residuals`` is the canonical stage name; the AHC handler
    # writes ``method='ahc'`` + ``linkage`` / ``metric`` /
    # ``distance_threshold`` / ``n_residuals`` / ``n_clusters`` /
    # ``n_noise``. ``auto_promote`` may carry ``cluster_count`` as a
    # secondary signal (number of clusters touched during propagation).
    for stage_name in ('cluster_residuals', 'auto_promote'):
        sd = stages.get(stage_name) or {}
        if not isinstance(sd, dict):
            continue
        if method is None and sd.get('method'):
            method = str(sd['method'])
        if cluster_count == 0 and 'n_clusters' in sd:
            cluster_count = int(sd.get('n_clusters') or 0)
        elif cluster_count == 0 and 'cluster_count' in sd:
            cluster_count = int(sd.get('cluster_count') or 0)
        if 'n_residuals' in sd:
            residual_count = int(sd.get('n_residuals') or residual_count)
        elif 'residual_count' in sd:
            residual_count = int(sd.get('residual_count') or residual_count)
        if 'n_noise' in sd:
            noise_count = int(sd.get('n_noise') or noise_count)
        elif 'noise_count' in sd:
            noise_count = int(sd.get('noise_count') or noise_count)

    return {
        'last_run_at': last_run_at,
        'cluster_count': cluster_count,
        'residual_count': residual_count,
        'noise_count': noise_count,
        'method': method,
    }


def _build_dataset_query_body(fields: RegionFields) -> dict[str, Any]:
    return {
        'size': 0,
        # track_total_hits=True so total_crops reflects the real count, not
        # the ES default 10000-hit cap. The dashboard prominently displays
        # this number — under-counting at 10k looks like a stuck pipeline.
        'track_total_hits': True,
        'aggs': {
            # --- legacy fields (preserved for back-compat) ----------------
            # source/class_source/region-detector/region-verifier are
            # all mapped keyword directly on the live index — no .keyword
            # subfield exists (only the region-status field is text+.keyword).
            'by_source': {'terms': {'field': 'source', 'size': 32}},
            'validated': {'filter': {'term': {'class_validated': True}}},
            'test_holdout': {'filter': {'term': {'test_holdout': True}}},
            # --- new: label provenance breakdown --------------------------
            'class_sources': {
                # missing: a terms agg silently drops docs with no
                # class_source.keyword value (e.g. an unlabel_crop'd
                # crop) instead of bucketing them — contradicts this
                # rollup's own "surface in 'other' rather than
                # dropping" intent and broke the labeled.* buckets'
                # sum-to-total_crops invariant on a real crop. Give
                # missing values an explicit bucket key that
                # _rollup_class_sources routes to 'other'.
                'terms': {
                    'field': 'class_source',
                    'size': 64,
                    'missing': '__none__',
                },
            },
            # The flat 'class_sources' agg above buckets EVERY crop by
            # class_source regardless of whether a class_id was ever
            # assigned -- 'vlm_unmatched' / 'vlm_new_class_pending' both
            # start with 'vlm' and class_id is null on both, so the old
            # labeled.by_vlm rollup (built straight off 'class_sources')
            # counted class-less crops as VLM-labeled. This sibling agg
            # scopes the same terms breakdown to docs that actually carry
            # a class_id, so ``labeled.*`` only counts real labels.
            'class_sources_with_class': {
                'filter': {'exists': {'field': 'class_id'}},
                'aggs': {
                    'by_source': {
                        'terms': {
                            'field': 'class_source',
                            'size': 64,
                            'missing': '__none__',
                        },
                    },
                },
            },
            # The class-less half of the same breakdown, so a VLM-touched
            # but never-classed crop (vlm_unmatched / vlm_new_class_pending)
            # can be surfaced explicitly under unlabeled.* instead of
            # silently vanishing from every bucket.
            'class_sources_no_class': {
                'filter': {'bool': {'must_not': [{'exists': {'field': 'class_id'}}]}},
                'aggs': {
                    'by_source': {
                        'terms': {
                            'field': 'class_source',
                            'size': 64,
                            'missing': '__none__',
                        },
                    },
                },
            },
            # region-detector breakdown — distinct from class_source. primary detector /
            # segmenter / human region detections show up here. The dashboard
            # surfaces "detector found N regions" from this, NOT from
            # class_source (which never carries a region-detector value).
            'region_detectors': {
                'terms': {'field': fields.detector, 'size': 16},
            },
            # region-verifier breakdown — VLM (AI) vs human. When the
            # operator hits 'Confirm' on the region-review page the
            # detector's bbox stays put (fields.detector unchanged) but
            # fields.verifier is set to 'human'. So this is the real
            # "human-confirmed region count", distinct from
            # fields.detector='human' which only fires when the operator
            # draws a brand-new bbox in the region editor.
            'region_verifiers': {
                'terms': {'field': fields.verifier, 'size': 16},
            },
            # Operator-touched regions: anything where the validated flag
            # is True AND a human was involved (either drew the bbox OR
            # confirmed an AI-proposed one). The dashboard surfaces this as
            # the honest "you confirmed N regions today" number.
            'regions_validated_by_human': {
                'filter': {
                    'bool': {
                        'filter': [{'term': {fields.validated: True}}],
                        'should': [
                            {'term': {fields.detector: 'human'}},
                            {'term': {fields.verifier: 'human'}},
                        ],
                        'minimum_should_match': 1,
                    }
                }
            },
            'region_status': {
                'terms': {'field': fields.status, 'size': 32},
            },
            # Crops that actually carry a region box right now. This — not
            # total_detected (which sums detector CREDIT, including
            # rejected/failed attempts) — is the honest "crops with a
            # region" number and matches the region cluster view.
            'region_boxed': {'filter': {'exists': {'field': fields.bbox_norm}}},
            'no_label_source': {
                'filter': {
                    'bool': {
                        'must_not': [{'exists': {'field': 'class_id'}}],
                    }
                }
            },
            # How many clusters the index holds now (noise ids < 0 are
            # not clusters).
            'distinct_clusters': {
                'filter': {'range': {'cluster_id': {'gte': 0}}},
                'aggs': {
                    'n': {'cardinality': {'field': 'cluster_id', 'precision_threshold': 4000}}
                },
            },
            # Negative cluster_id is reserved for noise (legacy HDBSCAN
            # convention; AHC doesn't emit -1 today but the agg stays so
            # any future hybrid algorithm still surfaces noise here).
            'noise_clusters': {
                'filter': {'range': {'cluster_id': {'lt': 0}}},
            },
        },
    }


@router.get('/stats/dataset')
async def stats_dataset(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Overall dataset balance + label-source + pipeline breakdown.

    Single ``_search?size=0`` with composite aggregations. Backwards-
    compatible: legacy callers still see ``total_crops``, ``validated``,
    ``test_holdout``, ``by_source``.

    New fields for the dashboard:

    - ``as_of`` — ISO8601 timestamp of the query.
    - ``labeled.{by_human, by_vlm, by_classifier, other}`` — rolled-up counts
      derived from ``class_source`` plus the ``class_validated`` /
      region-validated flags.
    - ``unlabeled.{pending_detection, pending_verification, no_label_source,
      vlm_no_class, by_proposal}`` — the region-status field plus crops with
      no ``class_id``. ``by_proposal`` is F-23's fixed accounting for
      detector-proposed-but-not-yet-classified crops (moved here from the
      always-0 ``labeled.by_proposal``, since those class_source values
      never carry a class_id).
    - ``in_progress.region_drain_total_unfinished`` — matches the value
      returned by ``/curation/ingest/region_drain``.
    - ``clusters.{last_run_at, cluster_count, residual_count, noise_count, method}`` —
      sourced from the persisted ``auto_label_job`` state when present;
      ``cluster_count`` falls back to live ``cluster_id`` cardinality.
    """
    fields = get_region_fields()
    body = _build_dataset_query_body(fields)
    try:
        # This is also the query the SSE stats stream re-runs
        # every ~10-15s (see pipeline_events.py's TTL cache). Shard
        # request-cache eligible (size:0, no `now`/random scoring) —
        # OpenSearch invalidates it on every index refresh anyway, so
        # this only helps (bursts of refresh-free polls hit cache) and
        # can't make results staler than they already are.
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body, request_cache=True)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc

    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    aggs = resp.get('aggregations') or {}

    # Rollup is built from ``class_sources_with_class`` (docs that
    # actually carry a class_id), not the flat ``class_sources`` agg --
    # otherwise a class-less vlm_unmatched/vlm_new_class_pending crop
    # counts as VLM-labeled. See the agg comment in
    # ``_build_dataset_query_body``.
    with_class_buckets = ((aggs.get('class_sources_with_class') or {}).get('by_source') or {}).get(
        'buckets'
    ) or []
    rollup = _rollup_class_sources(with_class_buckets)

    no_class_buckets = ((aggs.get('class_sources_no_class') or {}).get('by_source') or {}).get(
        'buckets'
    ) or []
    vlm_no_class = _sum_prefixed(
        {str(b.get('key', '')): int(b.get('doc_count', 0)) for b in no_class_buckets},
        VLM_CLASS_SOURCE,
    )
    # F-23: was miscounted as (always-zero) labeled.by_proposal; these
    # sources never carry a class_id, so the real count only exists among
    # the class-less buckets.
    by_proposal = _count_by_proposal(no_class_buckets)

    # Region-detector rollup — separate from class label rollup.
    # by_detector counts crops where the profile's primary region detector
    # found the region; by_segmenter where its secondary segmenter did.
    # region_total is the denominator for "% of crops with a region detection".
    profile = region_profile_or_neutral()
    region_detector_buckets: dict[str, int] = {}
    for b in (aggs.get('region_detectors') or {}).get('buckets') or []:
        region_detector_buckets[str(b.get('key', ''))] = int(b.get('doc_count', 0))
    regions_by_detector = _sum_prefixed(region_detector_buckets, profile.detector_model)
    regions_by_segmenter = _sum_prefixed(region_detector_buckets, profile.segmenter_name)
    regions_by_human_drew = _sum_prefixed(region_detector_buckets, profile.human_detector_name)
    region_total_detected = sum(region_detector_buckets.values())

    # region-verifier rollup — distinct attribution of who confirmed the
    # region, regardless of who detected the bbox. A region is verified
    # either by a human or by the VLM (whose verifier value is the VLM's
    # model id, so "not human" is the only deployment-neutral test).
    region_verifier_buckets: dict[str, int] = {}
    for b in (aggs.get('region_verifiers') or {}).get('buckets') or []:
        region_verifier_buckets[str(b.get('key', ''))] = int(b.get('doc_count', 0))
    regions_verified_by_human = _sum_prefixed(region_verifier_buckets, profile.human_detector_name)
    regions_verified_by_vlm = sum(region_verifier_buckets.values()) - regions_verified_by_human

    # Validated-by-human union (drew the bbox OR confirmed an AI bbox).
    # This is the honest "you reviewed N regions" count for the dashboard.
    regions_validated_by_human = int(
        (aggs.get('regions_validated_by_human') or {}).get('doc_count', 0)
    )

    region_status_buckets: dict[str, int] = {}
    for b in (aggs.get('region_status') or {}).get('buckets') or []:
        region_status_buckets[str(b.get('key', ''))] = int(b.get('doc_count', 0))

    pending_detection = region_status_buckets.get(RegionStatus.PENDING_DETECTION, 0)
    pending_verification = region_status_buckets.get(RegionStatus.PENDING_VERIFICATION, 0)
    region_drain_total_unfinished = pending_detection + pending_verification

    # V-1: same stall-reason computation GET /ingest/region_drain serves,
    # mirrored here so the dashboard's "In-flight pipeline" panel (which
    # reads this endpoint, not region_drain) can render *why* the queue
    # isn't shrinking instead of a bare "0 (stalled)".
    from src.services.curation.region_dependency_health import (
        check_region_dependencies,
        stall_reason as _region_stall_reason,
    )
    from src.services.triton_control import TritonControlService

    region_deps = await check_region_dependencies(
        TritonControlService().get_repository_index, profile
    )
    region_stall_reason = _region_stall_reason(region_deps, pending_detection=pending_detection)

    no_label_source = int((aggs.get('no_label_source') or {}).get('doc_count', 0))

    cluster_meta = _read_auto_label_clusters_meta(
        fallback_residual=int((aggs.get('noise_clusters') or {}).get('doc_count', 0)),
        fallback_noise=int((aggs.get('noise_clusters') or {}).get('doc_count', 0)),
    )
    # The last run's own count (often a residual pass) is not the index's
    # total, so serve both under explicit names.
    cluster_meta['last_run_cluster_count'] = cluster_meta['cluster_count'] or None
    cluster_meta['cluster_count'] = int(
        ((aggs.get('distinct_clusters') or {}).get('n') or {}).get('value', 0)
    )

    return {
        'as_of': _now_iso(),
        # --- legacy keys (do NOT remove — labeler getStats() reads these) -
        'total_crops': int(total),
        'validated': (aggs.get('validated') or {}).get('doc_count', 0),
        'test_holdout': (aggs.get('test_holdout') or {}).get('doc_count', 0),
        'by_source': (aggs.get('by_source') or {}).get('buckets', []),
        # --- new dashboard fields -----------------------------------------
        'labeled': {
            # Class-label provenance (denominator = total_crops).
            # by_human counts class_source startswith 'human' — the actual
            # "human labeled the class" signal. Auto-validation (majority
            # agreement) lives in by_classifier, not by_human.
            **rollup,
        },
        # Region-detection provenance (denominator = total_crops).
        #
        # - ``by_detector`` / ``by_segmenter`` / ``by_human_drew`` are the
        #   *detector* counts (who created the bbox). by_human_drew is
        #   the strict 'operator drew a new bbox from scratch' count.
        # - ``verified_by_human`` / ``verified_by_vlm`` are the
        #   *verifier* counts (who said 'yes that's a region').
        # - ``validated_by_human`` is the union — every region the
        #   operator touched, whether they drew the bbox or confirmed an
        #   AI-proposed one. This is the honest 'I reviewed N regions'
        #   number the dashboard surfaces to the operator.
        'regions': {
            # boxed = crops with a region bbox right now (the honest
            # "crops with a region" count). confirmed = the pipeline said
            # it's a real region (region status == 'detected').
            # total_detected sums detector CREDIT and includes
            # rejected/failed attempts, so it overstates real regions —
            # kept for back-compat but no longer the headline number.
            'boxed': int((aggs.get('region_boxed') or {}).get('doc_count', 0)),
            'confirmed': region_status_buckets.get(RegionStatus.DETECTED, 0),
            'total_detected': region_total_detected,
            'by_detector': regions_by_detector,
            'by_segmenter': regions_by_segmenter,
            # Kept under the legacy name so older labeler bundles keep
            # rendering something; new label is ``by_human_drew``.
            'by_human': regions_by_human_drew,
            'by_human_drew': regions_by_human_drew,
            'verified_by_human': regions_verified_by_human,
            'verified_by_vlm': regions_verified_by_vlm,
            'validated_by_human': regions_validated_by_human,
        },
        'unlabeled': {
            'pending_detection': pending_detection,
            'pending_verification': pending_verification,
            'no_label_source': no_label_source,
            # Crops a VLM answered or proposed but that never got a
            # class_id (sources "vlm_unmatched" / "vlm_new_class_pending")
            # -- these used to be double counted as VLM labeled in the
            # "labeled" section above even though they carry no class.
            # They are a subset of no_label_source, surfaced explicitly so
            # a dashboard can tell "no class_source at all" apart from
            # "the VLM tried but didn't land on a registry class".
            'vlm_no_class': vlm_no_class,
            # F-23: the detector proposed this crop as an object of
            # interest but nothing has classified it yet -- moved here
            # from labeled.by_proposal, which was structurally always 0
            # (these class_source values never carry a class_id, so they
            # could never appear in the class_id-scoped rollup that fed
            # it). A subset of no_label_source, same as vlm_no_class.
            'by_proposal': by_proposal,
        },
        'in_progress': {
            'region_drain_total_unfinished': region_drain_total_unfinished,
            # V-1: null when nothing is pending or every region-profile
            # Triton dependency is READY; otherwise a human-readable line
            # naming which dependency is down and since when.
            'region_stall_reason': region_stall_reason,
        },
        'clusters': cluster_meta,
    }
