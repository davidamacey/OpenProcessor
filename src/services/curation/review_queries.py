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
from src.services.curation.class_sources import VLM_CLASS_SOURCES
from src.services.curation.ingest_class_sources import (
    classifier_class_sources,
    unlabeled_proposal_class_sources,
)
from src.services.curation.training_cohorts import LOW_CONFIDENCE_MAX
from src.services.curation.vlm_class_attempt import VLM_CLASS_EMPTY_REASON_FIELD


KNOWN_TABS: tuple[str, ...] = (
    'all',
    'mismatches',
    'vlm_low_conf',
    'outliers',
    'uncertainty',
    'model_disagreements',
    'regions',
    'primary_low_conf',
    'classifier_blind_spots',
    'new_class_proposals',
)

# W0: a display label + description per tab, so the frontend stops
# hardcoding them (F7's "Classifier blind spots" for classifier_blind_spots in
# particular -- the id itself is renamed in a later wave). Served by
# ``GET {prefix}/review/tabs``.
TAB_LABELS: dict[str, tuple[str, str]] = {
    'all': ('All', 'Unified queue: every crop a human should look at, most-uncertain first'),
    'mismatches': ('Mismatches', "The VLM's reply did not match any registry class"),
    'vlm_low_conf': (
        'VLM low confidence',
        "The VLM's own confidence in its label is medium or low",
    ),
    'outliers': ('Outliers', 'Far from cluster centroid'),
    'uncertainty': ('Uncertainty', 'High active-learning probe entropy'),
    'model_disagreements': (
        'Model disagreements',
        'Validated crops where the new model disagrees with the human label',
    ),
    'regions': ('Regions', 'Region detected — needs human confirmation'),
    'primary_low_conf': (
        'Primary low confidence',
        'Largest subject — classifier unsure or missed',
    ),
    'classifier_blind_spots': (
        'Classifier blind spots',
        'Detector proposed an item the classifier missed entirely',
    ),
    'new_class_proposals': (
        'New class proposals',
        'Needs a class the registry does not have yet',
    ),
}


# Query filters every tab honours (``GET /review/{tab}`` and its locate
# twin). ``build_tab_query`` / ``build_review_request`` read this same
# table, so the served catalog can never advertise a filter a tab ignores.
COMMON_FILTERS: tuple[str, ...] = (
    'include_test',
    'max_rank',
    'min_blur_ratio',
    'min_mistakenness',
    'hide_near_duplicates',
    'class_id',
    'source',
    'conf_min',
    'conf_max',
)
# Tab-only filters, on top of COMMON_FILTERS.
TAB_EXTRA_FILTERS: dict[str, tuple[str, ...]] = {'regions': ('text', 'region_status')}
# A filter value a tab applies when the client omits it (DQ-M6: the two
# "primary subject" tabs are rank-limited by definition).
PRIMARY_SUBJECT_MAX_RANK = 2
TAB_FILTER_DEFAULTS: dict[str, dict[str, Any]] = {
    'primary_low_conf': {'max_rank': PRIMARY_SUBJECT_MAX_RANK},
    'classifier_blind_spots': {'max_rank': PRIMARY_SUBJECT_MAX_RANK},
    'regions': {'region_status': 'all'},
}

# The ``region_status`` filter's selectable values on the ``regions`` tab
# (DQ-B2 follow-up: a verifier-rejected candidate is reviewable but was
# unreachable from the queue). ``'all'`` is the default -- today's
# accepted-but-unvalidated boxes plus a rejected candidate that still has
# a box to show. Served through ``FILTER_SPECS`` below.
REGION_STATUS_FILTER_OPTIONS: tuple[dict[str, str], ...] = (
    {'value': 'all', 'label': 'All (accepted + rejected candidates)'},
    {'value': RegionStatus.DETECTED.value, 'label': 'Detected only'},
    {
        'value': RegionStatus.VERIFY_REJECTED.value,
        'label': 'Verifier-rejected candidates only',
    },
)
REGION_STATUS_FILTER_VALUES: frozenset[str] = frozenset(
    o['value'] for o in REGION_STATUS_FILTER_OPTIONS
)
# Self-describing specs for filters with a fixed, enumerable value set,
# keyed by query parameter. Served as ``filter_specs`` on
# ``GET /review/tabs`` so the frontend renders any enum filter generically.
FILTER_SPECS: dict[str, dict[str, Any]] = {
    'region_status': {
        'param': 'region_status',
        'kind': 'enum',
        'label': 'Status',
        'options': REGION_STATUS_FILTER_OPTIONS,
    },
}


def tab_filters(tab: str) -> tuple[str, ...]:
    """Query filters ``tab`` applies, in a stable order."""
    return (*COMMON_FILTERS, *TAB_EXTRA_FILTERS.get(tab, ()))


def review_tab_catalog() -> list[dict[str, Any]]:
    """``[{id, label, description, filters, filter_defaults,
    filter_specs}, ...]`` for every ``KNOWN_TABS`` entry.

    ``filters`` lists the query parameters the tab honours (anything else
    is accepted but ignored); ``filter_defaults`` the value a tab applies
    when that parameter is omitted (``{}`` for none); ``filter_specs`` a
    self-describing ``{param, kind, label, options: [{value, label}]}``
    entry for each of those filters that has a fixed enum (``[]`` for a
    tab with none), so the frontend renders it without per-filter code.

    Fails loudly (``KeyError``) if a tab is added to ``KNOWN_TABS`` without
    a matching ``TAB_LABELS`` entry -- the same "one source of truth"
    contract ``test_class_sources.py`` enforces for ``class_source``.
    """
    return [
        {
            'id': tab,
            'label': TAB_LABELS[tab][0],
            'description': TAB_LABELS[tab][1],
            'filters': list(tab_filters(tab)),
            'filter_defaults': dict(TAB_FILTER_DEFAULTS.get(tab, {})),
            'filter_specs': [
                {**FILTER_SPECS[name], 'options': [dict(o) for o in FILTER_SPECS[name]['options']]}
                for name in tab_filters(tab)
                if name in FILTER_SPECS
            ],
        }
        for tab in KNOWN_TABS
    ]


def mismatch_reason(src: dict[str, Any], registry_names: frozenset[str], default: str) -> str:
    """Per-item reason on the ``mismatches`` tab (DQ-m4).

    ``vlm_unmatched`` covers more than "the VLM named something outside
    the registry": a low-confidence answer that *is* a registry class is
    routed here unapplied, and legacy rows carry no answer at all. The
    tab's generic reason was false for both. ``registry_names`` are the
    normalized active class names.
    """
    from src.services.curation.new_class_terms import normalize_term

    raw = str(src.get('vlm_raw_class') or src.get('vlm_raw_label') or '').strip()
    if not raw:
        return 'VLM gave no class answer'
    if normalize_term(raw) in registry_names:
        confidence = src.get('vlm_confidence') or 'unknown'
        return f'VLM named registry class {raw!r} at {confidence} confidence; not applied'
    return default


def region_reason(src: dict[str, Any], fields: Any, default: str) -> str:
    """Per-item reason on the ``regions`` tab (mirrors :func:`mismatch_reason`).

    A verifier-rejected candidate needs a different reason than an
    accepted-but-unreviewed box: the reviewer is confirming/reversing a
    rejection, not just validating a fresh detection. Includes
    ``region_rejection_reason`` when the worker recorded one.
    """
    if src.get(fields.status) != RegionStatus.VERIFY_REJECTED.value:
        return default
    why = src.get(fields.rejection_reason)
    if why:
        return f'verifier rejected this candidate ({why}) — needs human review'
    return 'verifier rejected this candidate — needs human review'


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
    region_status: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Return ``(must, must_not, reason)`` for one review tab.

    ``region_status`` (``regions`` tab only, ignored elsewhere): one of
    :data:`REGION_STATUS_FILTER_VALUES`. Raises ``HTTPException(400, ...)``
    for an unrecognized ``tab`` or an unrecognized ``region_status`` — same
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
                        # The VLM was asked and gave no class.
                        {'exists': {'field': VLM_CLASS_EMPTY_REASON_FIELD}},
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
        # DQ-M8: select on the VLM's own confidence only. `confidence` is
        # the detector/classifier score, not the VLM's -- gating on it hid
        # every VLM-unsure item on a confidently-detected crop. The
        # class_source clause keeps a stale vlm_confidence (the label has
        # since been rewritten by a classifier or a human) out.
        must.append({'terms': {'vlm_confidence': ['medium', 'low']}})
        must.append({'terms': {'class_source': sorted(VLM_CLASS_SOURCES)}})
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
        # Region-detection review queue: every accepted region box a human
        # has not validated yet, including the worker's auto-confirmed ones
        # (``region_auto_confirmed`` is machine agreement, not validation)
        # -- PLUS (default / 'all') a verifier-rejected candidate that still
        # has a box to show (``region_candidate_bbox_norm``). Before this,
        # a rejected candidate had no bbox_norm, so it could never match
        # `exists bbox_norm` and was unreachable from this queue even
        # though the confirm-promotes-candidate write path already
        # supported reversing it (DQ-B2 follow-up). The reviewer opens
        # each in the region editor, adjusts the box if needed, and
        # confirms, which sets the human-only validated flag (a rejected
        # candidate's confirm promotes it into `bbox_norm` instead).
        region_status_filter = region_status or 'all'
        if region_status_filter not in REGION_STATUS_FILTER_VALUES:
            raise HTTPException(
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
        rejected_candidate = {
            'bool': {
                'filter': [
                    {'term': {fields.status: RegionStatus.VERIFY_REJECTED}},
                    {'exists': {'field': fields.candidate_bbox_norm}},
                ]
            }
        }
        if region_status_filter == RegionStatus.DETECTED.value:
            must.append({'exists': {'field': fields.bbox_norm}})
            must_not.append({'term': {fields.status: RegionStatus.NO_REGION_VISIBLE}})
            must_not.append({'term': {fields.status: RegionStatus.VERIFY_REJECTED}})
            must_not.append({'term': {fields.status: RegionStatus.FALSE_POSITIVE}})
            reason = 'region detected — needs human confirmation'
        elif region_status_filter == RegionStatus.VERIFY_REJECTED.value:
            must.append(rejected_candidate)
            reason = 'verifier rejected this candidate — needs human review'
        else:
            # 'all': today's accepted-but-unvalidated boxes, plus a
            # rejected candidate that still has a box to show. A
            # false_positive keeps its box too (`bbox_norm` stays set —
            # see RegionFields.candidate_bbox_norm's docstring) but is
            # terminal and must never re-enter the human queue.
            must.append(
                {
                    'bool': {
                        'should': [{'exists': {'field': fields.bbox_norm}}, rejected_candidate],
                        'minimum_should_match': 1,
                    }
                }
            )
            must_not.append({'term': {fields.status: RegionStatus.NO_REGION_VISIBLE}})
            must_not.append({'term': {fields.status: RegionStatus.FALSE_POSITIVE}})
            reason = 'region detected — needs human confirmation'
        # Substring search on region text, case-insensitive: stored case
        # depends on whichever writer set the text, so don't assume an
        # uppercase canonical form.
        if text and 'text' in tab_filters(tab):
            must.append(region_text_clause(fields.text, text))
        # Default sort: region score desc (falling back to the rejected
        # candidate's score when there is no accepted region score), so
        # the high-confidence items are reviewed first — see
        # review_sorts.py's two-key 'region_score' clause.
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
        reason = 'needs a class the registry does not have yet'
    else:
        raise HTTPException(
            status_code=400,
            detail=f'unknown review tab: {tab}. Must be one of: {", ".join(KNOWN_TABS)}',
        )

    # DQ-M6: subject-size limit, applied once here for every tab that
    # serves it (it used to be honoured only by the two primary tabs).
    rank_limit = (
        max_rank if max_rank is not None else TAB_FILTER_DEFAULTS.get(tab, {}).get('max_rank')
    )
    if rank_limit is not None and 'max_rank' in tab_filters(tab):
        must.append({'range': {'crop_rank_in_image': {'lte': rank_limit}}})

    return must, must_not, reason


__all__ = [
    'COMMON_FILTERS',
    'FILTER_SPECS',
    'KNOWN_TABS',
    'REGION_STATUS_FILTER_OPTIONS',
    'REGION_STATUS_FILTER_VALUES',
    'TAB_EXTRA_FILTERS',
    'TAB_FILTER_DEFAULTS',
    'TAB_LABELS',
    'build_tab_query',
    'mismatch_reason',
    'region_reason',
    'review_tab_catalog',
    'tab_filters',
]
