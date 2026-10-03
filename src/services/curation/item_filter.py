"""The one item filter: every list, search, stats and "act on a selection" route
builds its query from :class:`ItemFilter` through :func:`item_filter_clauses`.

A filter is a conjunction of predicates over stored items; a list inside one
predicate is an OR. Class identity is by NAME (``class_names`` matches an
item's ``class_name`` or the detector's ``proposal_name``, see
:mod:`~src.services.curation.name_match`), never by model class id.

``origin`` says how an item came to exist: ``detector`` (the ingest detector),
``sam3`` (a full-image SAM 3 pass), ``human`` (a person owns its class) or
``import`` (a dataset import wrote or proposed it). ``embedding_state``
``embedded`` means "has a vector" (the authoritative ``exists`` test); the
others are the recorded reasons it has none.

``max_rank`` keeps the N largest boxes per image (``crop_rank_in_image`` is the
box's area rank among the items stored on its image).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.services.curation.crop_browse import classifier_low_confidence_clause, confidence_band
from src.services.curation.embedding_state import EMBEDDED, EmbeddingState, embedded_clause
from src.services.curation.item_text import item_text_query
from src.services.curation.name_match import name_clause


Origin = Literal['detector', 'sam3', 'human', 'import']
ReviewStatus = Literal['pending', 'validated', 'dismissed', 'excluded']

HUMAN_CLASS_SOURCES: tuple[str, ...] = ('human', 'human_move')
# The item field a full-image SAM 3 pass stamps on the items it writes.
SAM3_DETECTOR = 'sam3'


class ItemFilter(BaseModel):
    """Predicates over stored items. Every field unset = no constraint."""

    model_config = ConfigDict(extra='forbid')

    class_names: list[str] = Field(default_factory=list)
    exclude_class_names: list[str] = Field(default_factory=list)
    conf_min: float | None = Field(default=None, ge=0.0, le=1.0)
    conf_max: float | None = Field(default=None, ge=0.0, le=1.0)
    # Box area as a fraction of its image.
    min_area: float | None = Field(default=None, ge=0.0, le=1.0)
    max_area: float | None = Field(default=None, ge=0.0, le=1.0)
    max_rank: int | None = Field(default=None, ge=1)
    origin: list[Origin] = Field(default_factory=list)
    embedding_state: list[EmbeddingState] = Field(default_factory=list)
    review_status: list[ReviewStatus] = Field(default_factory=list)
    # Ingest source tag (not the same thing as ``origin``).
    source: str | None = None
    class_id: int | None = None
    cluster_id: int | None = None
    import_id: str | None = None
    dataset_split: str | None = None
    label_source: str | None = None
    class_source: str | None = None
    label_validated: bool | None = None
    proposed_by_import: bool | None = None
    on_negative_frame: bool | None = None
    needs_new_class: bool | None = None
    review_dismissed: bool | None = None
    min_blur_ratio: float | None = Field(default=None, ge=0.0)
    classifier_conf_lt: float | None = Field(default=None, ge=0.0, le=1.0)
    item_text: str | None = Field(default=None, max_length=200)

    def is_empty(self) -> bool:
        return self == type(self)()


def _present(field: str, wanted: bool) -> dict[str, Any]:
    exists: dict[str, Any] = {'exists': {'field': field}}
    return exists if wanted else {'bool': {'must_not': exists}}


def _flag(field: str, wanted: bool) -> dict[str, Any]:
    marked: dict[str, Any] = {'term': {field: True}}
    return marked if wanted else {'bool': {'must_not': marked}}


def _any_of(clauses: list[dict[str, Any]]) -> dict[str, Any]:
    return (
        clauses[0]
        if len(clauses) == 1
        else {'bool': {'should': clauses, 'minimum_should_match': 1}}
    )


def _origin_clause(origin: str) -> dict[str, Any]:
    if origin == 'import':
        return {'exists': {'field': 'import_ids'}}
    if origin == 'human':
        return {'terms': {'class_source': list(HUMAN_CLASS_SOURCES)}}
    if origin == 'sam3':
        return {'term': {'detector': SAM3_DETECTOR}}
    # detector: everything no other origin claims
    return {
        'bool': {'must_not': [_origin_clause(o) for o in ('import', 'human', 'sam3')]},
    }


def _review_clause(status: str) -> dict[str, Any]:
    if status == 'validated':
        return {'term': {'class_validated': True}}
    if status == 'dismissed':
        return {'exists': {'field': 'review_dismissed_at'}}
    if status == 'excluded':
        return {'term': {'class_excluded': True}}
    return {  # pending: not decided by anyone and not set aside
        'bool': {
            'must_not': [
                {'term': {'class_validated': True}},
                {'exists': {'field': 'review_dismissed_at'}},
                {'term': {'class_excluded': True}},
            ]
        }
    }


def _embedding_clause(state: str) -> dict[str, Any]:
    if state == EMBEDDED:
        return embedded_clause()
    return {'term': {'embedding_state': state}}


def item_filter_clauses(f: ItemFilter) -> list[dict[str, Any]]:
    """The filter as AND-ed (filter-context) clauses.

    Raises ``ValueError`` for a malformed band (``conf_min`` above
    ``conf_max``, ``min_area`` above ``max_area``) or an ``item_text`` with no
    letter or digit, which the routes map to a 400.
    """
    out: list[dict[str, Any]] = []
    names = name_clause(f.class_names)
    if names is not None:
        out.append(names)
    excluded = name_clause(f.exclude_class_names)
    if excluded is not None:
        out.append({'bool': {'must_not': [excluded]}})
    band = confidence_band(f.conf_min, f.conf_max)
    if band is not None:
        out.append(band)
    if f.min_area is not None or f.max_area is not None:
        if f.min_area is not None and f.max_area is not None and f.min_area > f.max_area:
            msg = f'min_area ({f.min_area}) is greater than max_area ({f.max_area})'
            raise ValueError(msg)
        bounds = {k: v for k, v in (('gte', f.min_area), ('lte', f.max_area)) if v is not None}
        out.append({'range': {'crop_area_norm': bounds}})
    if f.max_rank is not None:
        out.append({'range': {'crop_rank_in_image': {'lte': f.max_rank}}})
    if f.origin:
        out.append(_any_of([_origin_clause(o) for o in f.origin]))
    if f.embedding_state:
        out.append(_any_of([_embedding_clause(s) for s in f.embedding_state]))
    if f.review_status:
        out.append(_any_of([_review_clause(s) for s in f.review_status]))
    for field, value in (
        ('source', f.source),
        ('class_id', f.class_id),
        ('cluster_id', f.cluster_id),
        ('dataset_split', f.dataset_split),
        ('label_source', f.label_source),
        ('class_source', f.class_source),
    ):
        if value is not None and value != '':
            out.append({'term': {field: value}})
    if f.import_id:
        out.append(
            _any_of(
                [
                    {'term': {'import_ids': f.import_id}},
                    {'term': {'proposed_by_import': f.import_id}},
                ]
            )
        )
    if f.label_validated is not None:
        out.append({'term': {'class_validated': f.label_validated}})
    if f.proposed_by_import is not None:
        out.append(_present('proposed_by_import', f.proposed_by_import))
    if f.on_negative_frame is not None:
        from src.services.curation.review_queries import negative_frame_clause

        out.append(negative_frame_clause(f.on_negative_frame))
    if f.needs_new_class is not None:
        out.append(_flag('needs_new_class', f.needs_new_class))
    if f.review_dismissed is not None:
        out.append(_present('review_dismissed_at', f.review_dismissed))
    if f.min_blur_ratio is not None:
        # An item with no blur score is not hidden by the clarity slider.
        out.append(
            {
                'bool': {
                    'should': [
                        {'range': {'blur_lap_ratio': {'gte': f.min_blur_ratio}}},
                        {'bool': {'must_not': {'exists': {'field': 'blur_lap_ratio'}}}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )
    if f.classifier_conf_lt is not None:
        out.append(classifier_low_confidence_clause(f.classifier_conf_lt))
    if f.item_text is not None:
        text = item_text_query(f.item_text)
        if text is None:
            msg = 'item_text must contain a letter or digit'
            raise ValueError(msg)
        out.append(text)
    return out


def visibility_clauses(*, include_test: bool, include_excluded: bool) -> list[dict[str, Any]]:
    """The browse defaults: hold-out items and human-ignored items are hidden
    unless asked for. Separate from the filter because only browse hides them."""
    out: list[dict[str, Any]] = []
    if not include_test:
        out.append({'bool': {'must_not': {'term': {'test_holdout': True}}}})
    if not include_excluded:
        out.append({'bool': {'must_not': {'term': {'class_excluded': True}}}})
    return out


__all__ = [
    'HUMAN_CLASS_SOURCES',
    'SAM3_DETECTOR',
    'ItemFilter',
    'Origin',
    'ReviewStatus',
    'item_filter_clauses',
    'visibility_clauses',
]
