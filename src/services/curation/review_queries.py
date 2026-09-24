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

from src.config.curation import ITEM_EMBEDDING_FIELD, PROBE_ENTROPY_REVIEW_MIN
from src.config.region_fields import get_region_fields
from src.config.region_state import RegionStatus
from src.services.curation.ingest_class_sources import (
    classifier_class_sources,
    unlabeled_proposal_class_sources,
)
from src.services.curation.training_cohorts import LOW_CONFIDENCE_MAX


KNOWN_TABS: tuple[str, ...] = (
    'all',
    'mismatches',
    'vlm_low_conf',
    'outliers',
    'uncertainty',
    'model_disagreements',
    'regions',
    'primary_low_conf',
    'coco_blind_spots',
    'new_class_proposals',
)


def _escape_wildcard(text: str) -> str:
    """Escape wildcard-query metacharacters so user text matches literally."""
    return text.replace('\\', '\\\\').replace('*', '\\*').replace('?', '\\?')


def region_text_clause(field: str, text: str) -> dict[str, Any]:
    """Substring match on ``field`` (typically :attr:`RegionFields.text`),
    case-insensitive and with user input escaped so wildcard metacharacters
    in the search string match literally (F-9). Stored case depends on
    whichever writer set the text, so this never assumes an uppercase
    canonical form -- unlike a naive ``f'*{text.upper()}*'`` wildcard, which
    is both case-sensitive against mixed-case stored values and vulnerable
    to a user-supplied ``*``/``?`` being interpreted as a wildcard.
    """
    return {
        'wildcard': {
            field: {
                'value': f'*{_escape_wildcard(text)}*',
                'case_insensitive': True,
            }
        }
    }


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
        # Items dismissed from review (POST /crops/{id}/discard with
        # dismiss_from_review, or the legacy review_dismiss) stay out of
        # every queue until undone / POST /crops/{id}/review_undismiss.
        {'exists': {'field': 'review_dismissed_at'}},
        # F-4: an excluded item (POST /crops/{id}/exclude) must never
        # reappear in any review tab, regardless of what else flags it.
        {'term': {'class_excluded': True}},
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
                        {'term': {'class_source': 'vlm_unmatched'}},
                        {'term': {'class_source': 'vlm_new_class_pending'}},
                        {'terms': {'vlm_confidence': ['medium', 'low']}},
                        {'range': {'cluster_distance': {'gte': 0.35}}},
                        # D-1 (F-6): `exists probe_pred_entropy` matches
                        # almost every non-holdout item after one probe
                        # run -- a no-op filter in practice. Gate on an
                        # actual uncertainty threshold instead.
                        {'range': {'probe_pred_entropy': {'gte': PROBE_ENTROPY_REVIEW_MIN}}},
                        # Crops with no class assigned at all (YOLO11 found a
                        # vehicle but neither the classifier nor the VLM got a usable label)
                        {
                            'bool': {
                                'must_not': [{'exists': {'field': 'class_id'}}],
                                'filter': [{'exists': {'field': ITEM_EMBEDDING_FIELD}}],
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
        # Crops where the VLM's class_source is vlm_unmatched, or where the classifier
        # disagreed with the VLM. The cheapest proxy: class_source == 'vlm'
        # AND confidence in {medium,low} → flagged because the classifier didn't match.
        must.append({'term': {'class_source': 'vlm_unmatched'}})
        reason = "VLM's reply did not match any registry class"
    elif tab == 'vlm_low_conf':
        must.append({'terms': {'vlm_confidence': ['medium', 'low']}})
        # Trust the classifier when it was very confident — sending those crops to the
        # human queue (with reason "VLM confidence below high") is noise.
        # Matches the classifier_confidence_skip_vlm=0.80 default in legacy_pipeline
        # and the _V6_LOW_CONF_THRESHOLD=0.80 in sam_worker/combined.py.
        must.append({'range': {'confidence': {'lt': 0.80}}})
        reason = 'VLM confidence below high'
    elif tab == 'outliers':
        # D-1 (F-6): outlier_flagged is never written anywhere in the
        # repo -- deleted. This queue is cluster_distance >= 0.35 only.
        must.append({'range': {'cluster_distance': {'gte': 0.35}}})
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
        must_not = [{'term': {'class_excluded': True}}]
        if not include_test:
            must_not.append({'term': {'test_holdout': True}})
        # Already auto-confirmed by the SAM worker — no human needed.
        # region-validated is set when LPR + Gemma (or SAM3 + Gemma) agree;
        # those two-AI-agreement crops should not enter the human queue.
        # SAM3-only crops (LPR missed) keep it false and remain
        # surfaced here — they are the LPR-training cohort.
        must_not.append({'term': {fields.validated: True}})
        # No class_validated exclusion: region review is independent of the
        # item's class. VLM and cluster agreement validate most classes
        # automatically, so excluding them hid nearly every unreviewed region.
        must_not.append({'term': {fields.status: RegionStatus.NO_REGION_VISIBLE}})
        must_not.append({'term': {fields.status: RegionStatus.VERIFY_REJECTED}})
        # Human already marked the detection a false positive (box kept
        # for FP analysis / LPR hard-negative training) — terminal, must
        # not re-enter the human queue.
        must_not.append({'term': {fields.status: RegionStatus.FALSE_POSITIVE}})
        # Substring search on region text, case-insensitive: stored case
        # depends on whichever writer set the text, so don't assume an
        # uppercase canonical form.
        if text:
            must.append(region_text_clause(fields.text, text))
        # Default sort: region score desc, so the high-confidence detections
        # are reviewed first (likely accept), low-score later (more
        # corrections expected) — see review_sorts.py.
        reason = 'region detected — needs human confirmation'
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
        must_not = [{'term': {'class_excluded': True}}]
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
        # Largest primary subjects where the classifier was unsure or never fired — the
        # highest-value labels for the next training pass. Confidence is a
        # band, not a floor: target everything below the 0.75 ingest floor
        # (or no classifier box at all) and let rank + the clarity slider strip the
        # junk, so a large clear crop the classifier whiffed on at 0.05 still surfaces.
        must.append({'range': {'crop_rank_in_image': {'lte': max_rank or 2}}})
        # D-1 (F-6): classifier_raw_confidence is never written in
        # production -- point the "unsure" branch at the stored
        # `confidence` field, restricted to items a classifier actually
        # scored (unlabeled_proposal_class_sources() below already covers
        # "no classifier box at all").
        must.append(
            {
                'bool': {
                    'should': [
                        {
                            'bool': {
                                'filter': [
                                    {'terms': {'class_source': sorted(classifier_class_sources())}},
                                    {'range': {'confidence': {'lt': LOW_CONFIDENCE_MAX}}},
                                ]
                            }
                        },
                        {'terms': {'class_source': sorted(unlabeled_proposal_class_sources())}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )
        # Default sort: 'primary_low_conf_default' — see review_sorts.py.
        reason = 'largest subject — classifier unsure or missed'
    elif tab == 'coco_blind_spots':
        # The cleanest blind spot: the item detector proposed an item that the classifier
        # missed entirely, on a primary subject. class_source is the exact
        # signal: an ingest proposal nothing classified. The stored
        # proposal score lives in ``confidence``.
        must.append({'terms': {'class_source': sorted(unlabeled_proposal_class_sources())}})
        must.append({'range': {'crop_rank_in_image': {'lte': max_rank or 2}}})
        # Default sort: 'coco_blind_spots_default' — see review_sorts.py.
        reason = 'detector proposed an item the classifier missed (blind spot)'
    elif tab == 'new_class_proposals':
        # Items that need a class the registry doesn't have yet: flagged by
        # a human (POST /crops/flag_new_class) or proposed by the VLM.
        must.append(
            {
                'bool': {
                    'should': [
                        {'term': {'needs_new_class': True}},
                        {'term': {'class_source': 'vlm_new_class_pending'}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )
        reason = 'needs a class the registry does not have yet'
    else:
        raise HTTPException(
            status_code=400,
            detail=f'unknown review tab: {tab}. Must be one of: {", ".join(KNOWN_TABS)}',
        )

    return must, must_not, reason


__all__ = ['KNOWN_TABS', 'build_tab_query']
