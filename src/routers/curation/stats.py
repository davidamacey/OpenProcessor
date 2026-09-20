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
    logger,
    router,
)


@router.get('/stats/classes')
async def stats_classes(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Per-class total/validated breakdown.

    Returns both the raw ``by_class`` aggregation and a flattened ``classes``
    array (``{class_id, class_name, count, validated_count}``) joined against
    the registry for the names. The labeler's ``getStats()`` (Export + Home
    pages) consumes ``classes`` — the aggregation alone has no class names, so
    without this join the Export dataset table renders empty.
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
    for bucket in aggs.get('by_class', {}).get('buckets', []):
        cid = int(bucket['key'])
        counts[cid] = int(bucket.get('doc_count', 0))
        validated[cid] = int((bucket.get('validated') or {}).get('doc_count', 0))

    classes: list[dict[str, Any]] = []
    try:
        reg = get_class_registry().load()
        classes = [
            {
                'class_id': c.class_id,
                'class_name': c.class_name,
                'count': counts.get(c.class_id, 0),
                'validated_count': validated.get(c.class_id, 0),
            }
            for c in reg.classes
            if not getattr(c, 'deprecated', False)
        ]
    except Exception as exc:
        logger.warning('stats_classes_registry_join_failed', error=str(exc))

    return {**aggs, 'classes': classes}


# Provenance keys that mean "human-applied or human-validated", used to
# avoid double-counting class_source buckets against the ``by_human``
# rollup which is already computed from class_validated / the region
# validated flag.
_HUMAN_SOURCE_PREFIXES = ('human',)
# Legacy CLIP-prototype provenance. Any rows that still carry it are
# bucketed under 'other'.
_LEGACY_PIPELINE_SOURCES = (
    'coco_yolo11',
    'coco_yolo11_proposal',
)


def _rollup_class_sources(buckets: list[dict[str, Any]]) -> dict[str, int]:
    """Roll up ``class_source`` buckets into dashboard taxonomy.

    Returns a dict of vehicle-class label provenance counts. ``by_human``
    counts ONLY rows whose ``class_source`` is a human-prefixed value
    (e.g. 'human', 'human_move'). A majority-agreement auto-validator
    also sets ``class_validated=True`` but that's auto-validation, not a
    human label — using the validated flag here would inflate by_human.
    Region-detector breakdown (e.g. lpr/sam3/vlm-ocr) is computed
    separately by the caller via region-detector aggregations.
    """
    by_human = 0
    by_vlm = 0
    by_v6 = 0
    by_yolo11_proposal = 0
    by_other = 0
    for b in buckets:
        key = str(b.get('key', ''))
        cnt = int(b.get('doc_count', 0))
        if key.startswith(_HUMAN_SOURCE_PREFIXES):
            by_human += cnt
        elif key.startswith(('gemma', 'vlm')):
            by_vlm += cnt
        elif key.startswith(('v6', 'cluster_v6')):
            by_v6 += cnt
        elif key.startswith(('coco_yolo11', 'yolo11')):
            # A generic detector proposed this crop as an object of
            # interest but the secondary classifier hasn't reached
            # majority confidence yet. These are awaiting-classification,
            # not "other unknown".
            by_yolo11_proposal += cnt
        else:
            # Truly unknown / future provenance — surface in 'other'
            # rather than dropping.
            by_other += cnt
    return {
        'by_human': by_human,
        'by_vlm': by_vlm,
        'by_v6': by_v6,
        'by_yolo11_proposal': by_yolo11_proposal,
        'other': by_other,
    }


def _read_auto_label_clusters_meta(
    fallback_residual: int,
    fallback_noise: int,
) -> dict[str, Any]:
    """Best-effort read of the auto_label_job persisted state file.

    Returns a dict with ``last_run_at``, ``cluster_count``,
    ``residual_count``, ``noise_count``, ``method``. Any read / parse
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
            # hdd_source/class_source/region-detector/region-verifier are
            # all mapped keyword directly on the live index — no .keyword
            # subfield exists (only the region-status field is text+.keyword).
            'by_source': {'terms': {'field': 'hdd_source', 'size': 32}},
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
            # region-detector breakdown — distinct from class_source. LPR /
            # SAM3 / human region detections show up here. The dashboard
            # surfaces "LPR found N regions" from this, NOT from
            # class_source (which never carries an lpr value).
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
                        'must': [{'term': {fields.validated: True}}],
                        'should': [
                            {'term': {fields.detector: 'human'}},
                            {'term': {fields.verifier: 'human'}},
                        ],
                        'minimum_should_match': 1,
                    }
                }
            },
            'region_status': {
                'terms': {'field': f'{fields.status}.keyword', 'size': 32},
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
            # cluster_id cardinality — fallback when we don't have a
            # persisted auto_label result.
            'distinct_clusters': {
                'cardinality': {'field': 'cluster_id', 'precision_threshold': 4000},
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
    - ``labeled.{by_human, by_vlm, by_v6, other}`` — rolled-up counts
      derived from ``class_source`` plus the ``class_validated`` /
      region-validated flags.
    - ``unlabeled.{pending_detection, pending_verification, no_label_source}`` —
      the region-status field plus crops with no ``class_id``.
    - ``in_progress.sam_drain_total_unfinished`` — matches the value
      returned by ``/curation/ingest/sam_drain``.
    - ``clusters.{last_run_at, cluster_count, residual_count, noise_count, method}`` —
      sourced from the persisted ``auto_label_job`` state when present;
      ``cluster_count`` falls back to live ``cluster_id`` cardinality.
    """
    fields = get_region_fields()
    body = _build_dataset_query_body(fields)
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch error: {exc}') from exc

    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    aggs = resp.get('aggregations') or {}

    rollup = _rollup_class_sources((aggs.get('class_sources') or {}).get('buckets') or [])

    # Region-detector rollup — separate from class label rollup. by_lpr
    # counts crops where an LPR-family detector found a region (NOT crops
    # with class_source='lpr', which never happens). region_total is the
    # denominator for "% of crops with a region detection".
    region_detector_buckets: dict[str, int] = {}
    for b in (aggs.get('region_detectors') or {}).get('buckets') or []:
        region_detector_buckets[str(b.get('key', ''))] = int(b.get('doc_count', 0))
    regions_by_lpr = sum(cnt for k, cnt in region_detector_buckets.items() if k.startswith('lpr'))
    regions_by_sam3 = sum(cnt for k, cnt in region_detector_buckets.items() if k.startswith('sam3'))
    regions_by_human_drew = sum(
        cnt for k, cnt in region_detector_buckets.items() if k.startswith('human')
    )
    region_total_detected = sum(region_detector_buckets.values())

    # region-verifier rollup — distinct attribution of who confirmed the
    # region, regardless of who detected the bbox.
    region_verifier_buckets: dict[str, int] = {}
    for b in (aggs.get('region_verifiers') or {}).get('buckets') or []:
        region_verifier_buckets[str(b.get('key', ''))] = int(b.get('doc_count', 0))
    regions_verified_by_human = sum(
        cnt for k, cnt in region_verifier_buckets.items() if k.startswith('human')
    )
    regions_verified_by_vlm = sum(
        cnt for k, cnt in region_verifier_buckets.items() if k.startswith(('gemma', 'vlm'))
    )

    # Validated-by-human union (drew the bbox OR confirmed an AI bbox).
    # This is the honest "you reviewed N regions" count for the dashboard.
    regions_validated_by_human = int(
        (aggs.get('regions_validated_by_human') or {}).get('doc_count', 0)
    )

    region_status_buckets: dict[str, int] = {}
    for b in (aggs.get('region_status') or {}).get('buckets') or []:
        region_status_buckets[str(b.get('key', ''))] = int(b.get('doc_count', 0))

    pending_detection = region_status_buckets.get(
        RegionStatus.PENDING_DETECTION, 0
    ) + region_status_buckets.get('pending', 0)
    pending_verification = region_status_buckets.get(
        RegionStatus.PENDING_VERIFICATION, 0
    ) + region_status_buckets.get('pending_verify', 0)
    sam_drain_total_unfinished = pending_detection + pending_verification

    no_label_source = int((aggs.get('no_label_source') or {}).get('doc_count', 0))

    cluster_meta = _read_auto_label_clusters_meta(
        fallback_residual=int((aggs.get('noise_clusters') or {}).get('doc_count', 0)),
        fallback_noise=int((aggs.get('noise_clusters') or {}).get('doc_count', 0)),
    )
    if cluster_meta['cluster_count'] == 0:
        cluster_meta['cluster_count'] = int((aggs.get('distinct_clusters') or {}).get('value', 0))

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
            # agreement) lives in by_v6, not by_human.
            **rollup,
        },
        # Region-detection provenance (denominator = total_crops).
        #
        # - ``by_lpr`` / ``by_sam3`` / ``by_human_drew`` are the
        #   *detector* counts (who created the bbox). by_human_drew is
        #   the strict 'operator drew a new bbox from scratch' count.
        # - ``verified_by_human`` / ``verified_by_vlm`` are the
        #   *verifier* counts (who said 'yes that's a region').
        # - ``validated_by_human`` is the union — every region the
        #   operator touched, whether they drew the bbox or confirmed an
        #   AI-proposed one. This is the honest 'I reviewed N regions'
        #   number the dashboard surfaces to the operator.
        'plates': {
            # boxed = crops with a region bbox right now (the honest
            # "crops with a region" count). confirmed = the pipeline said
            # it's a real region (region status == 'detected').
            # total_detected sums detector CREDIT and includes
            # rejected/failed attempts, so it overstates real regions —
            # kept for back-compat but no longer the headline number.
            'boxed': int((aggs.get('region_boxed') or {}).get('doc_count', 0)),
            'confirmed': region_status_buckets.get(RegionStatus.DETECTED, 0),
            'total_detected': region_total_detected,
            'by_lpr': regions_by_lpr,
            'by_sam3': regions_by_sam3,
            # Kept under the legacy name so older labeler bundles keep
            # rendering something; new label is ``by_human_drew``.
            'by_human': regions_by_human_drew,
            'by_human_drew': regions_by_human_drew,
            'verified_by_human': regions_verified_by_human,
            'verified_by_gemma': regions_verified_by_vlm,
            'validated_by_human': regions_validated_by_human,
        },
        'unlabeled': {
            'pending_detection': pending_detection,
            'pending_verification': pending_verification,
            'no_label_source': no_label_source,
        },
        'in_progress': {
            'sam_drain_total_unfinished': sam_drain_total_unfinished,
        },
        'clusters': cluster_meta,
    }
