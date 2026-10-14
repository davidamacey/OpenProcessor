"""Per-tab query construction for ``GET /curation/review/{tab}``.

Split out of the review router verbatim (no behavior change) so that
router's sort/filter additions could land without breaching the
700-LOC pre-commit ceiling on ``src/routers/curation/*.py``
(``.pre-commit-config.yaml``'s ``max-file-size`` hook). This module owns
*what a tab matches* (``must``/``must_not`` + the human-readable ``reason``);
``review_sorts.py`` owns *what order results come back in* — the two are
deliberately independent registries, so a tab's match logic can
never accidentally couple to its default sort.
"""

from __future__ import annotations

from typing import Any

from src.config.curation import ITEM_EMBEDDING_FIELD, PROBE_ENTROPY_REVIEW_MIN
from src.config.region_fields import get_region_fields
from src.config.region_state import RegionStatus
from src.config.vlm_class_attempt import VLM_CLASS_EMPTY_REASON_FIELD
from src.core.exceptions import RequestRejectedError
from src.domain.embedding_state import embedded_clause
from src.domain.ingest_class_sources import (
    LABEL_IMPORT_CLASS_SOURCE,
    classifier_class_sources,
    unlabeled_proposal_class_sources,
)
from src.domain.region_boxes import box_query
from src.services.curation import review_tabs
from src.services.curation.class_sources import VLM_CLASS_SOURCES
from src.services.curation.review_filter_specs import (
    HAS_REJECTED_BOX,
    REGION_STATUS_DEFAULT,
    REGION_STATUS_FILTER_VALUES,
)
from src.services.curation.training_cohorts import LOW_CONFIDENCE_MAX
from src.services.curation.vlm_class_attempt import EmptyClassReason


# Both fields are keyword-mapped (see the model_disagreements note below), so the
# comparison is plain doc-value equality. ``detector_class_name`` is stored in
# registry-name form (item_doc.DETECTOR_FIELDS), the same form as ``class_name``.
DETECTOR_DISAGREES_SCRIPT = (
    "doc.containsKey('detector_class_name') && doc['detector_class_name'].size() > 0 && "
    "doc.containsKey('class_name') && doc['class_name'].size() > 0 && "
    "!doc['detector_class_name'].value.equals(doc['class_name'].value)"
)


def negative_frame_clause(on_negative_frame: bool) -> dict[str, Any]:
    """The one definition of the ``on_negative_frame`` filter, shared by
    ``GET /crops`` and the review queue."""
    marked: dict[str, Any] = {'term': {'on_negative_frame': True}}
    return marked if on_negative_frame else {'bool': {'must_not': marked}}


def _escape_wildcard(text: str) -> str:
    """Escape wildcard-query metacharacters so user text matches literally."""
    return text.replace('\\', '\\\\').replace('*', '\\*').replace('?', '\\?')


def region_text_clause(field: str, text: str) -> dict[str, Any]:
    """Substring match on ``field`` (typically :attr:`RegionFields.text`),
    case-insensitive and with user input escaped so wildcard metacharacters
    in the search string match literally. Stored case depends on
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
    region_status: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Return ``(must, must_not, reason)`` for one review tab.

    ``region_status`` (``regions`` tab only, ignored elsewhere): one of
    :data:`REGION_STATUS_FILTER_VALUES`. Raises ``RequestRejectedError(400, ...)``
    for an unrecognized ``tab`` or an unrecognized ``region_status`` — same
    behavior the review router had inline before this split.
    """
    fields = get_region_fields()
    must: list[dict[str, Any]] = []
    must_not: list[dict[str, Any]] = [
        {'term': {'class_validated': True}},
        # Items dismissed from review (POST /crops/{id}/discard with
        # dismiss_from_review, or the legacy review_dismiss) stay out of
        # every queue until undone / POST /crops/{id}/review_undismiss.
        {'exists': {'field': 'review_dismissed_at'}},
        # An excluded item (POST /crops/{id}/exclude) must never
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
                        # A project combine kept one source's label where another
                        # disagreed (owner D6): a human decides.
                        {'term': {'combine_conflict': True}},
                        # The VLM was asked and gave no class.
                        {'exists': {'field': VLM_CLASS_EMPTY_REASON_FIELD}},
                        {'terms': {'vlm_confidence': ['medium', 'low']}},
                        {'range': {'cluster_distance': {'gte': 0.35}}},
                        # `exists probe_pred_entropy` matches
                        # almost every non-holdout item after one probe
                        # run -- a no-op filter in practice. Gate on an
                        # actual uncertainty threshold instead.
                        {'range': {'probe_pred_entropy': {'gte': PROBE_ENTROPY_REVIEW_MIN}}},
                        # Crops with no class assigned at all (the detector found an
                        # item but neither the classifier nor the VLM got a usable label)
                        {
                            'bool': {
                                'must_not': [{'exists': {'field': 'class_id'}}],
                                'filter': [embedded_clause(ITEM_EMBEDDING_FIELD)],
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
        # Select on the VLM's own confidence only. `confidence` is
        # the detector/classifier score, not the VLM's -- gating on it hid
        # every VLM-unsure item on a confidently-detected crop. The
        # class_source clause keeps a stale vlm_confidence (the label has
        # since been rewritten by a classifier or a human) out.
        must.append({'terms': {'vlm_confidence': ['medium', 'low']}})
        must.append({'terms': {'class_source': sorted(VLM_CLASS_SOURCES)}})
        reason = 'VLM confidence below high'
    elif tab == 'detector_disagreements':
        # The VLM labelled the crop with a class other than the one the detector
        # gave it. Unvalidated only (the default must_not): a human decision
        # already settled the rest. A VLM answer outside the registry has no
        # class_name and belongs to the mismatches tab, not here.
        must.append({'terms': {'class_source': sorted(VLM_CLASS_SOURCES)}})
        must.append({'exists': {'field': 'detector_class_name'}})
        must.append({'exists': {'field': 'class_name'}})
        must.append(
            {'script': {'script': {'source': DETECTOR_DISAGREES_SCRIPT, 'lang': 'painless'}}}
        )
        # Default sort: 'detector_disagreement_default' — see review_sorts.py.
        reason = "VLM's class differs from the detector's class"
    elif tab == 'outliers':
        # outlier_flagged is never written anywhere in the
        # repo -- deleted. This queue is cluster_distance >= 0.35 only.
        must.append({'range': {'cluster_distance': {'gte': 0.35}}})
        # Default sort: 'atypicality' — see review_sorts.py.
        reason = 'outlier — far from cluster centroid'
    elif tab == 'uncertainty':
        # The active-learning probe writes ``probe_pred_entropy`` per
        # crop. High entropy = the v7-nano probe is uncertain. Sort desc.
        must.append({'exists': {'field': 'probe_pred_entropy'}})
        # Default sort: 'uncertainty_entropy' — see review_sorts.py.
        reason = 'high probe entropy — active-learning candidate'
    elif tab == 'regions':
        # Region-detection review queue: every accepted region box a human
        # has not validated yet, including the worker's auto-confirmed ones
        # (``region_auto_confirmed`` is machine agreement, not validation)
        # -- PLUS (default / 'all') a verifier-rejected candidate that still
        # has a box to show (``region_candidate_bbox_norm``). Before this,
        # a rejected candidate had no bbox_norm, so it could never match
        # `exists bbox_norm` and was unreachable from this queue even
        # though the confirm-promotes-candidate write path already
        # supported reversing it. The reviewer opens
        # each in the region editor, adjusts the box if needed, and
        # confirms, which sets the human-only validated flag (a rejected
        # candidate's confirm promotes it into `bbox_norm` instead).
        region_status_filter = region_status or REGION_STATUS_DEFAULT
        if region_status_filter not in REGION_STATUS_FILTER_VALUES:
            raise RequestRejectedError(
                status_code=400,
                detail=(
                    f'unknown region_status filter: {region_status_filter!r}. '
                    f'Must be one of: {", ".join(sorted(REGION_STATUS_FILTER_VALUES))}'
                ),
            )
        must_not = [{'term': {'class_excluded': True}}]
        if not include_test:
            must_not.append({'term': {'test_holdout': True}})
        # Human-validated regions are done.
        must_not.append({'term': {fields.validated: True}})
        # No class_validated exclusion: region review is independent of the
        # item's class. VLM and cluster agreement validate most classes
        # automatically, so excluding them hid nearly every unreviewed region.
        #
        # `region_bbox_norm` / `region_candidate_bbox_norm` are
        # retired -- a box is now `accepted` (the old "has bbox_norm")
        # or `rejected` (the old "has a kept candidate") inside the item's
        # `region_boxes` list, checked via a nested query.
        has_accepted_box = box_query(
            {'term': {f'{fields.boxes}.{fields.boxes_state}': 'accepted'}}, fields
        )
        # `verify_rejected` is "every box rejected": the item status plus at
        # least one rejected box (an item with no box at all never matches).
        rejected_candidate = {
            'bool': {
                'filter': [
                    {'term': {fields.status: RegionStatus.VERIFY_REJECTED}},
                    {'range': {fields.rejected_count: {'gte': 1}}},
                ]
            }
        }
        if region_status_filter == RegionStatus.DETECTED.value:
            must.append(has_accepted_box)
            must_not.append({'term': {fields.status: RegionStatus.NO_REGION_VISIBLE}})
            must_not.append({'term': {fields.status: RegionStatus.VERIFY_REJECTED}})
            must_not.append({'term': {fields.status: RegionStatus.FALSE_POSITIVE}})
            reason = 'region detected — needs human confirmation'
        elif region_status_filter == RegionStatus.VERIFY_REJECTED.value:
            must.append(rejected_candidate)
            reason = 'verifier rejected this candidate — needs human review'
        elif region_status_filter == HAS_REJECTED_BOX:
            must.append({'range': {fields.rejected_count: {'gte': 1}}})
            reason = 'a box was rejected — needs human review'
        else:
            # 'all': today's accepted-but-unvalidated boxes, plus a
            # rejected candidate that still has a box to show. A
            # false_positive keeps its box too (its state stays
            # 'false_positive', not 'accepted' -- see
            # RegionStatus.FALSE_POSITIVE's docstring) but is terminal and
            # must never re-enter the human queue.
            must.append(
                {
                    'bool': {
                        'should': [has_accepted_box, rejected_candidate],
                        'minimum_should_match': 1,
                    }
                }
            )
            must_not.append({'term': {fields.status: RegionStatus.NO_REGION_VISIBLE}})
            must_not.append({'term': {fields.status: RegionStatus.FALSE_POSITIVE}})
            reason = 'region detected — needs human confirmation'
        # Substring search on region text, case-insensitive: stored case
        # depends on whichever writer set the text, so don't assume an
        # uppercase canonical form. Per-box now (the fixed 'text' element
        # key, not RegionFields-indirected -- see region_boxes.py).
        if text and 'text' in review_tabs.tab_filters(tab):
            must.append(box_query(region_text_clause(f'{fields.boxes}.text', text), fields))
        # Default sort: region score desc (falling back to the rejected
        # candidate's score when there is no accepted region score), so
        # the high-confidence items are reviewed first — see
        # review_sorts.py's two-key 'region_score' clause.
    elif tab == 'model_disagreements':
        # Active-learning loop: after a
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
        # This was invisible until the probe backfill gave
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
                        # the clusters router's top_class agg for the full story
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
        # classifier_raw_confidence is never written in
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
    elif tab == 'classifier_blind_spots':
        # The cleanest blind spot: the item detector proposed an item that the classifier
        # missed entirely, on a primary subject. class_source is the exact
        # signal: an ingest proposal nothing classified. The stored
        # proposal score lives in ``confidence``.
        must.append({'terms': {'class_source': sorted(unlabeled_proposal_class_sources())}})
        # Default sort: 'classifier_blind_spots_default' — see review_sorts.py.
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
        # Two ways a stale/mismatched flag lands an item here that
        # doesn't need a new class at all:
        #  1. the VLM attempt gave no answer at all (empty_reason ==
        #     no_answer) -- nothing was proposed, so it can't need a new
        #     class for a term that doesn't exist;
        #  2. the item already carries a resolved class_id -- either a
        #     later write (resolve/reclassify/human label) settled it and
        #     left needs_new_class stale, or the human flag was set on an
        #     already-classed item. Either way "needs a new class" no
        #     longer describes it once it has one.
        must_not.append({'term': {VLM_CLASS_EMPTY_REASON_FIELD: EmptyClassReason.NO_ANSWER.value}})
        must_not.append({'exists': {'field': 'class_id'}})
        reason = 'needs a class the registry does not have yet'
    elif tab == 'imported':
        # Spot-check queue for labels a dataset import wrote: validated by
        # construction (the user asserts them as ground truth), so this tab
        # inverts the default "not validated" rule. A dismissed or excluded
        # item stays out; a frozen test split stays out unless include_test.
        must.append({'term': {'class_source': LABEL_IMPORT_CLASS_SOURCE}})
        must.append({'term': {'class_validated': True}})
        must_not = [
            {'exists': {'field': 'review_dismissed_at'}},
            {'term': {'class_excluded': True}},
        ]
        if not include_test:
            must_not.append({'term': {'test_holdout': True}})
        reason = 'imported label: spot check'
    else:
        raise RequestRejectedError(
            status_code=400,
            detail=f'unknown review tab: {tab}. Must be one of: {", ".join(review_tabs.KNOWN_TABS)}',
        )

    # Subject-size limit, applied once here for every tab that
    # serves it (it used to be honoured only by the two primary tabs).
    rank_limit = (
        max_rank
        if max_rank is not None
        else review_tabs.TAB_FILTER_DEFAULTS.get(tab, {}).get('max_rank')
    )
    if rank_limit is not None and 'max_rank' in review_tabs.tab_filters(tab):
        must.append({'range': {'crop_rank_in_image': {'lte': rank_limit}}})

    return must, must_not, reason
