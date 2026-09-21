"""Per-tab query construction for ``GET /curation/review/{tab}`` (curation-strategy
plan §7 Phase 3 / §10.6).

Split out of the review router verbatim (no behavior change) so that
router's Phase 3 sort/filter additions could land without breaching the
700-LOC pre-commit ceiling on ``src/routers/curation/*.py``
(``.pre-commit-config.yaml``'s ``max-file-size`` hook). This module owns
*what a tab matches* (``must``/``must_not`` + the human-readable ``reason``);
``review_sorts.py`` owns *what order results come back in* — the two are
deliberately independent registries (plan §0), so a tab's match logic can
never accidentally couple to its default sort.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.config.region_fields import get_region_fields
from src.config.region_state import RegionStatus


KNOWN_TABS: tuple[str, ...] = (
    'all',
    'mismatches',
    'gemma_low_conf',
    'outliers',
    'uncertainty',
    'model_disagreements',
    'regions',
    'primary_low_conf',
    'coco_blind_spots',
)


def build_tab_query(
    tab: str,
    *,
    include_test: bool,
    text: str | None,
    max_rank: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Return ``(must, must_not, reason)`` for one review tab.

    Raises ``HTTPException(400, ...)`` for an unrecognized ``tab`` — same
    behavior ``legacy_review.py`` had inline before this split.
    """
    fields = get_region_fields()
    must: list[dict[str, Any]] = []
    must_not: list[dict[str, Any]] = [
        {'term': {'class_validated': True}},
        # Crops the operator explicitly dismissed via /review's Discard
        # button stay out of every queue forever (until an /undismiss is
        # added). The Discard handler at DELETE /curation/crops/{id}/label
        # stamps ``review_dismissed_at``.
        {'exists': {'field': 'review_dismissed_at'}},
    ]
    if not include_test:
        must_not.append({'term': {'test_holdout': True}})
    reason: str

    if tab == 'all':
        # Unified review queue — every crop a human should look at, with the
        # most-uncertain ones first. Avoids the "click through 4 tabs"
        # workflow: just keep scrolling. Each item still carries a `reason`
        # field that explains which signal flagged it.
        must.append(
            {
                'bool': {
                    'should': [
                        {'term': {'class_source': 'gemma_unmatched'}},
                        {'term': {'class_source': 'gemma_new_class_pending'}},
                        {'terms': {'gemma_confidence': ['medium', 'low']}},
                        # NOTE: outlier_flagged is never written anywhere in the repo — permanent no-op; see below.
                        {'term': {'outlier_flagged': True}},
                        {'range': {'cluster_distance': {'gte': 0.35}}},
                        {'exists': {'field': 'probe_pred_entropy'}},
                        # Crops with no class assigned at all (YOLO11 found a
                        # vehicle but neither v6 nor Gemma got a usable label)
                        {
                            'bool': {
                                'must_not': [{'exists': {'field': 'class_id'}}],
                                'must': [{'exists': {'field': 'embedding'}}],
                            }
                        },
                    ],
                    'minimum_should_match': 1,
                },
            }
        )
        # Default sort: 'atypicality' (outliers first, highest cluster_distance,
        # then unsorted) — see review_sorts.py.
        reason = 'needs human review'
    elif tab == 'mismatches':
        # Crops where Gemma's class_source is gemma_unmatched, or where v6
        # disagreed with Gemma. The cheapest proxy: class_source == 'gemma'
        # AND confidence in {medium,low} → flagged because v6 didn't match.
        must.append({'term': {'class_source': 'gemma_unmatched'}})
        reason = "gemma's reply did not match any registry class"
    elif tab == 'gemma_low_conf':
        must.append({'terms': {'gemma_confidence': ['medium', 'low']}})
        # Trust v6 when it was very confident — sending those crops to the
        # human queue (with reason "gemma confidence below high") is noise.
        # Matches the v6_confidence_skip_gemma=0.80 default in legacy_pipeline
        # and the _V6_LOW_CONF_THRESHOLD=0.80 in sam_worker/combined.py.
        must.append({'range': {'confidence': {'lt': 0.80}}})
        reason = 'gemma confidence below high'
    elif tab == 'outliers':
        # NOTE: outlier_flagged is never written anywhere in the repo, so this queue is effectively cluster_distance >= 0.35 only.
        must.append(
            {
                'bool': {
                    'should': [
                        {'term': {'outlier_flagged': True}},
                        {'range': {'cluster_distance': {'gte': 0.35}}},
                    ],
                    'minimum_should_match': 1,
                },
            }
        )
        # Default sort: 'atypicality' — see review_sorts.py.
        reason = 'outlier — far from cluster centroid'
    elif tab == 'uncertainty':
        # Wave 5 active learning probe writes ``probe_pred_entropy`` per
        # crop. High entropy = the v7-nano probe is uncertain. Sort desc.
        must.append({'exists': {'field': 'probe_pred_entropy'}})
        # Default sort: 'uncertainty_entropy' — see review_sorts.py.
        reason = 'high probe entropy — active-learning candidate'
    elif tab == 'regions':
        # Region-detection review queue. Surfaces crops where the detector /
        # segmenter / VLM chain produced a bounding box that was NOT
        # auto-confirmed — i.e. the segmenter score, VLM confidence, or bbox
        # shape didn't all clear the worker's auto-confirm thresholds. The
        # user opens each in the region editor,
        # tweaks the bbox if needed, and clicks Confirm; that flips
        # label_validated=true. High-confidence triple-agreement crops are
        # already ``label_validated=true`` and skip this queue entirely.
        must.append({'exists': {'field': fields.bbox_norm}})
        must_not = []
        if not include_test:
            must_not.append({'term': {'test_holdout': True}})
        # Already auto-confirmed by the SAM worker — no human needed.
        # region-validated is set when LPR + Gemma (or SAM3 + Gemma) agree;
        # those two-AI-agreement crops should not enter the human queue.
        # SAM3-only crops (LPR missed) keep it false and remain
        # surfaced here — they are the LPR-training cohort.
        must_not.append({'term': {fields.validated: True}})
        must_not.append({'term': {'class_validated': True}})
        must_not.append({'term': {f'{fields.status}.keyword': RegionStatus.NO_REGION_VISIBLE}})
        must_not.append({'term': {f'{fields.status}.keyword': RegionStatus.VERIFY_REJECTED}})
        # Human already marked the detection a false positive (box kept
        # for FP analysis / LPR hard-negative training) — terminal, must
        # not re-enter the human queue.
        must_not.append({'term': {f'{fields.status}.keyword': RegionStatus.FALSE_POSITIVE}})
        # F5 — let the labeler search by region text on the review queue.
        # region text is a keyword field so a case-insensitive substring
        # search uses wildcard on the uppercase form (the worker stores
        # text canonicalized to upper).
        if text:
            must.append({'wildcard': {fields.text: f'*{text.upper()}*'}})
        # Default sort: region score desc, so the high-confidence detections
        # are reviewed first (likely accept), low-score later (more
        # corrections expected) — see review_sorts.py.
        reason = 'plate detected — needs human confirmation'
    elif tab == 'model_disagreements':
        # Active-learning loop (design §15.6 / 17 Phase 5): after a
        # promote, /curation/pipeline/auto_label re-scores crops with the new
        # model and writes probe_pred_class. This tab surfaces validated
        # crops where the new model's prediction differs from the human
        # label — high-signal: either re-label, or feed back into the
        # next training cycle.
        must.append({'term': {'class_validated': True}})
        must.append({'exists': {'field': 'probe_pred_class'}})
        # Override the default must_not — we WANT validated crops here.
        must_not = []
        if not include_test:
            must_not.append({'term': {'test_holdout': True}})
        # Inequality requires a script — the index is small enough at
        # typical curation-deployment scale (~163K items) that script_query
        # latency is fine.
        # ``.keyword`` sub-fields, not the bare ``text`` fields: both
        # probe_pred_class and class_name are mapped `text` with fielddata
        # disabled (the OpenSearch/ES default), so `doc['probe_pred_class']`
        # throws "Fielddata is disabled on text fields" at query time.
        # This was invisible until Phase 11's probe backfill gave
        # probe_pred_class its first real (non-empty) coverage — before
        # that, the `exists: probe_pred_class` clause matched zero docs, so
        # the script never ran against a real document. Confirmed live
        # against triton-opensearch (script_exception /
        # illegal_argument_exception before this fix).
        must.append(
            {
                'script': {
                    'script': {
                        # Both fields are mapped `keyword` directly on the
                        # live index — no `.keyword` subfield exists (see
                        # legacy_clusters.py's top_class agg for the full story
                        # on why this repo's code assumed one).
                        'source': (
                            "doc.containsKey('probe_pred_class') && "
                            "doc['probe_pred_class'].size() > 0 && "
                            "doc.containsKey('class_name') && "
                            "doc['class_name'].size() > 0 && "
                            "!doc['probe_pred_class'].value.equals("
                            "doc['class_name'].value)"
                        ),
                        'lang': 'painless',
                    }
                }
            }
        )
        # Default sort: 'disagreement_entropy_asc' — most-confident
        # disagreements first, so the user can quickly distinguish "model is
        # right, human was wrong" cases — see review_sorts.py.
        reason = 'new model disagrees with the validated label'
    elif tab == 'primary_low_conf':
        # Largest primary subjects where v6 was unsure or never fired — the
        # highest-value labels for the next training pass. Confidence is a
        # band, not a floor: target everything below the 0.75 ingest floor
        # (or no v6 box at all) and let rank + the clarity slider strip the
        # junk, so a large clear crop v6 whiffed on at 0.05 still surfaces.
        must.append({'range': {'crop_rank_in_image': {'lte': max_rank or 2}}})
        must.append(
            {
                'bool': {
                    'should': [
                        {'range': {'v6_raw_confidence': {'lt': 0.75}}},
                        {'bool': {'must_not': {'exists': {'field': 'v6_raw_confidence'}}}},
                        {'term': {'class_source': 'v6_low_conf'}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )
        must_not.append({'term': {'class_excluded': True}})
        # Default sort: 'primary_low_conf_default' — see review_sorts.py.
        reason = 'largest subject — v6 unsure or missed'
    elif tab == 'coco_blind_spots':
        # The cleanest blind spot: COCO YOLO11 detected a vehicle that v6
        # missed entirely, on a primary subject. class_source is the exact
        # signal (ingest restricts COCO proposals to vehicle classes). The
        # stored COCO detection score lives in ``confidence``.
        must.append({'term': {'class_source': 'coco_yolo11_proposal'}})
        must.append({'range': {'crop_rank_in_image': {'lte': max_rank or 2}}})
        must_not.append({'term': {'class_excluded': True}})
        # Default sort: 'coco_blind_spots_default' — see review_sorts.py.
        reason = 'COCO found a vehicle v6 missed (blind spot)'
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f'unknown review tab: {tab}. '
                'Must be one of: all, mismatches, gemma_low_conf, outliers, '
                'uncertainty, model_disagreements, regions, primary_low_conf, '
                'coco_blind_spots'
            ),
        )

    return must, must_not, reason


__all__ = ['KNOWN_TABS', 'build_tab_query']
