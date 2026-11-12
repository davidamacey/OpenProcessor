"""Review queues: the queue registry, labels, served filters and the catalog.

Owns *which queues exist and what each one advertises* (``GET {prefix}/review/tabs``);
:mod:`~src.services.curation.review_queries` owns what a tab matches and
:mod:`~src.services.curation.review_reasons` the per-item reason text.
"""

from __future__ import annotations

from typing import Any

from src.services.curation import review_filter_specs
from src.services.curation.review_filters import REGION_STATUS_DEFAULT


# The queues, in the order the rail shows them. Every other id in
# ``review_tab_clauses.SIMPLE_TABS`` is a ``review_tab`` filter value only, never a path.
QUEUES: tuple[str, ...] = (
    'to_label',
    'check_machine',
    'disagreements',
    'new_class_proposals',
    'audit',
    'regions',
)

# A display label + description per queue, served by ``GET {prefix}/review/tabs``.
QUEUE_LABELS: dict[str, tuple[str, str]] = {
    'to_label': ('To label', 'Items with no class yet, the most confident detections first'),
    'check_machine': (
        'Check machine labels',
        'Classes a machine set and no person has confirmed, least confident first',
    ),
    'disagreements': (
        'Disagreements',
        (
            'The VLM reply matched no class, the VLM and the detector name different classes, '
            'or the new model disagrees with a validated label'
        ),
    ),
    'new_class_proposals': (
        'New class proposals',
        'Needs a class the registry does not have yet',
    ),
    'audit': ('Audit sample', 'Crops drawn for the accuracy audit that still need a human label'),
    'regions': ('Regions', 'Region detected, needs human confirmation'),
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
    'source',
    'conf_min',
    'conf_max',
    'class_name',
    'exclude_class_name',
    'min_area',
    'max_area',
    'origin',
    'embedding_state',
    'review_status',
    'combine_conflict',
    'on_negative_frame',
    'taken_from',
    'taken_to',
)
# Tab-only filters, on top of COMMON_FILTERS.
TAB_EXTRA_FILTERS: dict[str, tuple[str, ...]] = {
    'regions': ('text', 'region_text_match', 'region_status'),
}
# A filter value a tab applies when the client omits it (the two "primary subject" ids, which are
# ``review_tab`` filter values, are rank-limited by definition).
PRIMARY_SUBJECT_MAX_RANK = 2
TAB_FILTER_DEFAULTS: dict[str, dict[str, Any]] = {
    'primary_low_conf': {'max_rank': PRIMARY_SUBJECT_MAX_RANK},
    'classifier_blind_spots': {'max_rank': PRIMARY_SUBJECT_MAX_RANK},
    'regions': {'region_status': REGION_STATUS_DEFAULT},
}


def tab_filters(tab: str) -> tuple[str, ...]:
    """Query filters ``tab`` applies, in a stable order. The ``regions``
    tab's ``text`` filter only exists on a text-reading region profile."""
    extra = TAB_EXTRA_FILTERS.get(tab, ())
    if 'text' in extra and not _region_text_enabled():
        extra = tuple(name for name in extra if name not in ('text', 'region_text_match'))
    return (*COMMON_FILTERS, *extra)


def _region_text_enabled() -> bool:
    from src.services.detection.profile_registry import get_active_region_profile

    profile = get_active_region_profile()
    return profile is None or profile.reads_text


def review_tab_catalog() -> list[dict[str, Any]]:
    """``[{id, label, description, filters, filter_defaults,
    filter_specs}, ...]`` for every ``QUEUES`` entry, EXCEPT
    ``regions`` when no region profile is active (no-profile gating
    contract: a client must not offer a queue for data that can never
    exist). The router also drops ``audit`` while no audit sample is pending.

    ``filters`` lists the query parameters the tab honours (anything else
    is accepted but ignored); ``filter_defaults`` the value a tab applies
    when that parameter is omitted (``{}`` for none); ``filter_specs`` a
    self-describing entry (``review_filter_specs``) for every one of those
    filters, so the frontend renders it without per-filter code.

    Fails loudly (``KeyError``) if a queue is added to ``QUEUES`` without
    a matching ``QUEUE_LABELS`` entry -- the same "one source of truth"
    contract ``test_class_sources.py`` enforces for ``class_source``.

    The ``regions`` tab's label/description come from the active region
    profile's ``display_name`` when one is configured (e.g. "Wheels"),
    falling back to the generic ``QUEUE_LABELS`` entry ("Regions") when it
    set no ``display_name``.
    """
    from src.services.detection.profile_registry import get_active_region_profile

    has_region_profile = get_active_region_profile() is not None
    return [
        {
            'id': tab,
            'label': _tab_label(tab),
            'description': _tab_description(tab),
            'filters': list(tab_filters(tab)),
            'filter_defaults': dict(TAB_FILTER_DEFAULTS.get(tab, {})),
            'filter_specs': [
                {
                    **review_filter_specs.FILTER_SPECS[name],
                    'options': [dict(o) for o in review_filter_specs.FILTER_SPECS[name]['options']],
                }
                for name in tab_filters(tab)
            ],
        }
        for tab in QUEUES
        if tab != 'regions' or has_region_profile
    ]


def _tab_label(tab: str) -> str:
    if tab == 'regions':
        from src.services.detection.profile_registry import get_active_region_profile

        profile = get_active_region_profile()
        if profile is not None and profile.display_name:
            return profile.display_name
    return QUEUE_LABELS[tab][0]


def _tab_description(tab: str) -> str:
    return QUEUE_LABELS[tab][1]
