"""Self-describing specs for every filter ``GET /review/{tab}`` honours.

Served as ``filter_specs`` on ``GET /review/tabs`` so a client renders any
filter (its widget, label, range and fixed options) without per-filter code.
:data:`FILTER_SPECS` has one entry per query parameter in
:data:`~src.services.curation.review_tabs.COMMON_FILTERS` and
``TAB_EXTRA_FILTERS`` (a test pins that), so a tab can never advertise a
filter with no spec.

Kinds: ``enum`` (pick one of ``options``), ``multi_enum`` (repeatable
parameter, any of ``options``), ``class_names`` (repeatable class NAME; no
fixed options: see its ``description`` for where the names come from),
``bool``, ``number`` / ``integer`` (``min``/``max``), ``text`` and ``date`` (an ISO
8601 day, ``YYYY-MM-DD``; the capture-date range is the two specs ``taken_from`` /
``taken_to``, both optional and inclusive).
"""

from __future__ import annotations

from typing import Any, get_args

from src.config.region_state import RegionStatus
from src.domain.embedding_state import EmbeddingState
from src.services.curation.item_filter import ItemFilter, Origin, ReviewStatus
from src.services.curation.region_text_search import TextMatch
from src.services.curation.review_filters import REGION_STATUS_DEFAULT, ReviewFilters


# The ``region_status`` filter's selectable values on the ``regions`` tab
# (a verifier-rejected candidate is reviewable but was
# unreachable from the queue). ``'all'`` is the default -- today's
# accepted-but-unvalidated boxes plus a rejected candidate that still has
# a box to show.
# Not a RegionStatus: it selects on the box list, whatever the item status
# (a `detected` item can carry a rejected box beside its accepted ones).
HAS_REJECTED_BOX = 'has_rejected_box'
REGION_STATUS_FILTER_OPTIONS: tuple[dict[str, str], ...] = (
    {'value': REGION_STATUS_DEFAULT, 'label': 'All (accepted + rejected boxes)'},
    {'value': RegionStatus.DETECTED.value, 'label': 'Detected only'},
    {
        'value': RegionStatus.VERIFY_REJECTED.value,
        'label': 'Items with only rejected boxes',
    },
    {'value': HAS_REJECTED_BOX, 'label': 'Items with any rejected box'},
)
REGION_STATUS_FILTER_VALUES: frozenset[str] = frozenset(
    o['value'] for o in REGION_STATUS_FILTER_OPTIONS
)
DATASET_SPLIT_FILTER_OPTIONS: tuple[dict[str, str], ...] = (
    {'value': 'train', 'label': 'Train'},
    {'value': 'val', 'label': 'Validation'},
    {'value': 'test', 'label': 'Test'},
)
ON_NEGATIVE_FRAME_FILTER_OPTIONS: tuple[dict[str, str], ...] = (
    {'value': 'true', 'label': 'Only items on a reviewed-negative frame'},
    {'value': 'false', 'label': 'Hide items on a reviewed-negative frame'},
)

# Selecting it means: omit the parameter (the filter is off).
UNSET_OPTION: dict[str, str] = {'value': '', 'label': 'Any'}

EMBEDDING_STATE_LABELS: dict[str, str] = {
    'embedded': 'Embedded',
    'not_selected': 'Not selected by the ingest policy',
    'deferred': 'Deferred: no vector yet by design',
    'failed': 'Embedding failed',
}
ORIGIN_LABELS: dict[str, str] = {
    'detector': 'Ingest detector',
    'sam3': 'Open-vocabulary pass',
    'human': 'A person',
    'drawn': 'Drawn by a person',
    'import': 'Dataset import',
}
TEXT_MATCH_LABELS: dict[str, str] = {
    'exact': 'Exact (letters and digits)',
    'fuzzy': 'Fuzzy (one edit)',
    'fragment': 'Contains fragment',
}
REVIEW_STATUS_LABELS: dict[str, str] = {
    'pending': 'Pending review',
    'validated': 'Validated',
    'dismissed': 'Dismissed from review',
    'excluded': 'Excluded',
}


def _options(values: tuple[str, ...], labels: dict[str, str]) -> tuple[dict[str, str], ...]:
    """One option per ``Literal`` value; a value without a label raises."""
    return tuple({'value': v, 'label': labels[v]} for v in values)


EMBEDDING_STATE_OPTIONS = _options(get_args(EmbeddingState), EMBEDDING_STATE_LABELS)

_CLASS_NAMES_HELP = (
    'Class NAMES, matched against an item class name and the detector label: take the '
    "registry names from GET /classes and the detector's labels from GET /ingest/config "
    '(detector.labels); case, spaces and hyphens are normalized.'
)


_TAKEN_HELP = (
    'Capture day (YYYY-MM-DD), inclusive; items with no capture time never match. '
    'taken_from must not be after taken_to.'
)

# Spec params whose filter-model field is named differently.
_MODEL_FIELD = {'class_name': 'class_names', 'exclude_class_name': 'exclude_class_names'}


def _model_default(param: str) -> Any:
    """The value applied when ``param`` is omitted, read off the filter models
    (``ReviewFilters`` first, then ``ItemFilter``) so a spec cannot drift."""
    field = _MODEL_FIELD.get(param, param)
    review = ReviewFilters()
    if hasattr(review, field):
        return getattr(review, field)
    item = ItemFilter()
    if hasattr(item, field):
        return getattr(item, field)
    raise KeyError(f'no filter-model field for spec param {param!r}')


def _spec(
    param: str,
    kind: str,
    label: str,
    *,
    options: tuple[dict[str, str], ...] = (),
    min_: float | None = None,
    max_: float | None = None,
    description: str = '',
) -> dict[str, Any]:
    default = _model_default(param)
    if kind == 'enum' and default is None:
        # The unset state is a visible choice, never an implied first option.
        options = (UNSET_OPTION, *options)
    return {
        'param': param,
        'kind': kind,
        'label': label,
        'options': options,
        'min': min_,
        'max': max_,
        'description': description,
        'default': default,
        'allows_unset': default is None,
    }


FILTER_SPECS: dict[str, dict[str, Any]] = {
    s['param']: s
    for s in (
        _spec('include_test', 'bool', 'Include test-holdout items'),
        _spec('max_rank', 'integer', 'Largest boxes per image', min_=1),
        _spec('min_blur_ratio', 'number', 'Minimum clarity', min_=0.0),
        _spec('min_mistakenness', 'number', 'Minimum mistakenness', min_=0.0),
        _spec('hide_near_duplicates', 'bool', 'Hide near-duplicates'),
        _spec('source', 'text', 'Ingest source tag'),
        _spec('conf_min', 'number', 'Confidence from', min_=0.0, max_=1.0),
        _spec('conf_max', 'number', 'Confidence to', min_=0.0, max_=1.0),
        _spec('class_name', 'class_names', 'Class', description=_CLASS_NAMES_HELP),
        _spec('exclude_class_name', 'class_names', 'Hide class', description=_CLASS_NAMES_HELP),
        _spec('min_area', 'number', 'Box size from (fraction of image)', min_=0.0, max_=1.0),
        _spec('max_area', 'number', 'Box size to (fraction of image)', min_=0.0, max_=1.0),
        _spec('origin', 'multi_enum', 'Origin', options=_options(get_args(Origin), ORIGIN_LABELS)),
        _spec('embedding_state', 'multi_enum', 'Embedding', options=EMBEDDING_STATE_OPTIONS),
        _spec(
            'review_status',
            'multi_enum',
            'Review status',
            options=_options(get_args(ReviewStatus), REVIEW_STATUS_LABELS),
        ),
        _spec('combine_conflict', 'bool', 'Project-combine conflicts only'),
        _spec(
            'on_negative_frame',
            'enum',
            'Negative frames',
            options=ON_NEGATIVE_FRAME_FILTER_OPTIONS,
        ),
        _spec('taken_from', 'date', 'Taken from', description=_TAKEN_HELP),
        _spec('taken_to', 'date', 'to', description=_TAKEN_HELP),
        _spec('text', 'text', 'Box text contains'),
        _spec(
            'region_text_match',
            'enum',
            'Text match',
            options=_options(get_args(TextMatch), TEXT_MATCH_LABELS),
            description='How the box text search matches; unset is a case-insensitive substring.',
        ),
        _spec('region_status', 'enum', 'Status', options=REGION_STATUS_FILTER_OPTIONS),
        _spec('import_id', 'text', 'Dataset import'),
        _spec('dataset_split', 'enum', 'Split', options=DATASET_SPLIT_FILTER_OPTIONS),
    )
}

__all__ = [
    'DATASET_SPLIT_FILTER_OPTIONS',
    'EMBEDDING_STATE_LABELS',
    'FILTER_SPECS',
    'HAS_REJECTED_BOX',
    'ON_NEGATIVE_FRAME_FILTER_OPTIONS',
    'REGION_STATUS_DEFAULT',
    'REGION_STATUS_FILTER_OPTIONS',
    'REGION_STATUS_FILTER_VALUES',
    'UNSET_OPTION',
]
