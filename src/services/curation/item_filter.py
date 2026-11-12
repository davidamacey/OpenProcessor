"""The one item filter: every list, search, stats and "act on a selection" route
builds its query from :class:`ItemFilter` through :func:`item_filter_clauses`.

A filter is a conjunction of predicates over stored items; a list inside one
predicate is an OR. Class identity is by NAME (``class_names`` matches an
item's ``class_name`` or the detector's ``proposal_name``, see
:mod:`~src.services.curation.name_match`), never by model class id.

``origin`` says how an item came to exist: ``detector`` (the ingest detector),
``sam3`` (a full-image SAM 3 pass), ``human`` (a person owns its class),
``drawn`` (a person drew its box) or ``import`` (a dataset import wrote or proposed it). ``embedding_state``
``embedded`` means "has a vector" (the authoritative ``exists`` test); the
others are the recorded reasons it has none.

``max_rank`` keeps the N largest boxes per image (``crop_rank_in_image`` is the
box's area rank among the items stored on its image).
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import get_region_fields
from src.domain.embedding_state import EMBEDDED, EmbeddingState, embedded_clause, state_clause
from src.domain.geo_box import GeoBox, check_polygon
from src.domain.item_text import item_text_query
from src.domain.region_boxes import box_query
from src.domain.viz_box import VizBox, check_viz_box
from src.services.curation.crop_browse import classifier_low_confidence_clause, confidence_band
from src.services.curation.image_join import ImageJoin, ImageJoinMissingError, has_image_predicates
from src.services.curation.name_match import DETECTOR_NAME_FIELDS, name_clause
from src.services.curation.region_text_search import TextMatch, region_text_match_clause
from src.services.detection.region_text import active_confusables


if TYPE_CHECKING:
    from src.services.embedding_spaces.readers import WriteFields


Origin = Literal['detector', 'sam3', 'human', 'import', 'drawn']
ReviewStatus = Literal['pending', 'validated', 'dismissed', 'excluded']

HUMAN_CLASS_SOURCES: tuple[str, ...] = ('human', 'human_move')
# Every item a full-image SAM 3 pass writes carries the prompt set it came from
# (and keeps it after a human relabels it), so that field is what ``sam3`` means.
SAM3_ORIGIN_FIELD = 'open_vocab_set'
# The most image or cluster ids one filter carries in a body; a query string stays at 200.
MAX_SELECTION_IDS = 5000
QUERY_IDS_MAX = 200


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
    # The items' clusters in the served partition (UUIDs, OR); the facet chips of one cluster open it.
    cluster_uids: list[uuid.UUID] = Field(default_factory=list, max_length=MAX_SELECTION_IDS)
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
    # Plate-text search on a box (`region_text_match_clause`): the item has a box whose folded
    # text matches.
    region_text: str | None = Field(default=None, max_length=200)
    region_text_match: TextMatch = 'exact'
    # The detector's own class (``detector_class_name``, by NAME like ``class_names``).
    detector_class_names: list[str] = Field(default_factory=list)
    # The detector's confidence: ``detector_conf_min`` inclusive, ``detector_conf_max``
    # exclusive, so adjacent bands (the cluster facets' histogram bins) never overlap.
    detector_conf_min: float | None = Field(default=None, ge=0.0, le=1.0)
    detector_conf_max: float | None = Field(default=None, ge=0.0, le=1.0)
    # A review tab id (one of ``review_tab_clauses.SIMPLE_TABS``): only the items that
    # tab's queue matches, the same clauses ``GET /review/{tab}`` runs.
    review_tab: str | None = None
    # Provenance of an item a full-image SAM 3 pass wrote.
    open_vocab_set: str | None = None
    source_prompt: str | None = None
    # True: only items the region gate skipped (``region_gate_skip`` set; the
    # segmenter was never asked, so they are safe to re-run).
    region_gate_skipped: bool | None = None
    # Capture date range, inclusive days. Compared with the UTC capture time when the offset is
    # known and with the wall-clock time (as if UTC) when it is not.
    taken_from: dt.date | None = None
    taken_to: dt.date | None = None
    # Only items on these images (the map opens the list on the images of one cell).
    image_ids: list[str] = Field(default_factory=list, max_length=MAX_SELECTION_IDS)
    # Predicates about the item's image. Items carry no coordinate, so these are resolved to the
    # matching images first (``ImageJoin``) and snapped to the project's location policy.
    geo_bbox: GeoBox | None = None
    geo_polygon: list[tuple[float, float]] | None = None
    has_location: bool | None = None
    near_dup_group: uuid.UUID | None = None
    # A box drawn on the embedding scatter: the items whose cached projection point is inside it.
    viz_box: VizBox | None = None

    @field_validator('viz_box')
    @classmethod
    def _viz_box_not_inverted(cls, box: VizBox | None) -> VizBox | None:
        return None if box is None else check_viz_box(box)

    @field_validator('geo_polygon')
    @classmethod
    def _polygon_ring(
        cls, vertices: list[tuple[float, float]] | None
    ) -> list[tuple[float, float]] | None:
        return None if vertices is None else check_polygon(vertices)

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
    if origin == 'drawn':
        # A person drew the box: human geometry on an item no detector ever named.
        return {
            'bool': {
                'filter': [{'term': {'geometry_source': 'human'}}],
                'must_not': [{'exists': {'field': 'detector_class_name'}}],
            }
        }
    if origin == 'sam3':
        return {'exists': {'field': SAM3_ORIGIN_FIELD}}
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


def _embedding_clause(state: str, embedding: WriteFields) -> dict[str, Any]:
    if state == EMBEDDED:
        return embedded_clause(embedding.vector_field)
    return state_clause(state, embedding.state_field)


def _name_clauses(f: ItemFilter) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    names = name_clause(f.class_names)
    if names is not None:
        out.append(names)
    excluded = name_clause(f.exclude_class_names)
    if excluded is not None:
        out.append({'bool': {'must_not': [excluded]}})
    return out


def _size_clauses(f: ItemFilter) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
    return out


def _detector_clauses(f: ItemFilter, embedding: WriteFields) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    names = name_clause(f.detector_class_names, DETECTOR_NAME_FIELDS)
    if names is not None:
        out.append(names)
    lo, hi = f.detector_conf_min, f.detector_conf_max
    if lo is not None and hi is not None and lo >= hi:
        msg = f'detector_conf_min ({lo}) must be below detector_conf_max ({hi})'
        raise ValueError(msg)
    bounds = {k: v for k, v in (('gte', lo), ('lt', hi)) if v is not None}
    if bounds:
        out.append({'range': {'detector_confidence': bounds}})
    if f.review_tab is not None:
        out.append(_review_tab_clause(f.review_tab, embedding))
    return out


def _review_tab_clause(tab: str, embedding: WriteFields) -> dict[str, Any]:
    from src.services.curation.review_queries import build_tab_query
    from src.services.curation.review_tab_clauses import SIMPLE_TABS

    if tab not in SIMPLE_TABS:
        msg = f'unknown review_tab {tab!r}; choose from {sorted(SIMPLE_TABS)}'
        raise ValueError(msg)
    must, must_not, _reason = build_tab_query(
        tab, include_test=False, text=None, max_rank=None, vector_field=embedding.vector_field
    )
    return {'bool': {'must': must, 'must_not': must_not}}


def _state_clauses(f: ItemFilter, embedding: WriteFields) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if f.origin:
        out.append(_any_of([_origin_clause(o) for o in f.origin]))
    if f.embedding_state:
        out.append(_any_of([_embedding_clause(s, embedding) for s in f.embedding_state]))
    if f.review_status:
        out.append(_any_of([_review_clause(s) for s in f.review_status]))
    return out


def _term_clauses(f: ItemFilter) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if f.image_ids:
        out.append({'terms': {'image_id': f.image_ids}})
    if f.cluster_uids:
        out.append({'terms': {'cluster_uid': [str(u) for u in f.cluster_uids]}})
    for field, value in (
        ('source', f.source),
        ('class_id', f.class_id),
        ('dataset_split', f.dataset_split),
        ('label_source', f.label_source),
        ('class_source', f.class_source),
        ('open_vocab_set', f.open_vocab_set),
        ('source_prompt', f.source_prompt),
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
    return out


def _flag_clauses(f: ItemFilter) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if f.proposed_by_import is not None:
        out.append(_present('proposed_by_import', f.proposed_by_import))
    if f.on_negative_frame is not None:
        from src.services.curation.review_queries import negative_frame_clause

        out.append(negative_frame_clause(f.on_negative_frame))
    if f.region_gate_skipped is not None:
        from src.config.region_fields import get_region_fields

        out.append(_present(get_region_fields().gate_skip, f.region_gate_skipped))
    if f.needs_new_class is not None:
        out.append(_flag('needs_new_class', f.needs_new_class))
    if f.review_dismissed is not None:
        out.append(_present('review_dismissed_at', f.review_dismissed))
    return out


def _capture_clauses(f: ItemFilter) -> list[dict[str, Any]]:
    if f.taken_from is None and f.taken_to is None:
        return []
    if f.taken_from is not None and f.taken_to is not None and f.taken_from > f.taken_to:
        msg = f'taken_from ({f.taken_from}) is after taken_to ({f.taken_to})'
        raise ValueError(msg)
    bounds: dict[str, str] = {}
    if f.taken_from is not None:
        bounds['gte'] = f'{f.taken_from.isoformat()}T00:00:00'
    if f.taken_to is not None:
        bounds['lt'] = f'{(f.taken_to + dt.timedelta(days=1)).isoformat()}T00:00:00'
    return [
        _any_of(
            [
                {'range': {'capture_taken_utc': bounds}},
                {
                    'bool': {
                        'must': [{'range': {'capture_taken_local': bounds}}],
                        'must_not': [{'exists': {'field': 'capture_taken_utc'}}],
                    }
                },
            ]
        )
    ]


def _quality_clauses(f: ItemFilter) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
    if f.region_text is not None:
        out.append(
            box_query(
                region_text_match_clause(
                    get_region_fields().boxes,
                    f.region_text,
                    f.region_text_match,
                    active_confusables(),
                )
            )
        )
    return out


def _image_clauses(f: ItemFilter, join: ImageJoin | None) -> list[dict[str, Any]]:
    """The image predicates as item clauses: the items on the joined images. A group alone also
    matches the items that carry the group id themselves."""
    if not has_image_predicates(f):
        return []
    if join is None:
        msg = 'the filter has image predicates (area, location, near-duplicate group) but no image join was passed'
        raise ImageJoinMissingError(msg)
    on_images: dict[str, Any] = {'terms': {'image_id': list(join.image_ids)}}
    only_group = f.near_dup_group is not None and not any(
        (f.geo_bbox, f.geo_polygon, f.has_location is not None)
    )
    if only_group:
        return [_any_of([{'term': {'near_dup_group_id': str(f.near_dup_group)}}, on_images])]
    return [on_images]


def _viz_clauses(f: ItemFilter) -> list[dict[str, Any]]:
    if f.viz_box is None:
        return []
    x1, y1, x2, y2 = f.viz_box
    return [
        {'range': {'viz_x': {'gte': x1, 'lte': x2}}},
        {'range': {'viz_y': {'gte': y1, 'lte': y2}}},
    ]


def item_filter_clauses(
    f: ItemFilter, *, embedding: WriteFields, image_join: ImageJoin | None = None
) -> list[dict[str, Any]]:
    """The filter as AND-ed (filter-context) clauses.

    ``image_join`` is the resolved image predicates (:func:`resolve_image_join`); a filter with
    image predicates and no join raises :class:`ImageJoinMissingError` instead of ignoring them.

    ``embedding`` are the primary item space's vector and state fields (``item_write_fields()``):
    the ``embedding_state`` predicate and the ``all`` review tab read them.

    Raises ``ValueError`` for a malformed band (``conf_min`` above
    ``conf_max``, ``min_area`` above ``max_area``) or an ``item_text`` or ``region_text`` with no
    letter or digit, which the routes map to a 400.
    """
    return [
        *_name_clauses(f),
        *_size_clauses(f),
        *_detector_clauses(f, embedding),
        *_state_clauses(f, embedding),
        *_term_clauses(f),
        *_flag_clauses(f),
        *_capture_clauses(f),
        *_quality_clauses(f),
        *_image_clauses(f, image_join),
        *_viz_clauses(f),
    ]


def item_filter_query(
    f: ItemFilter, *, embedding: WriteFields, image_join: ImageJoin | None = None
) -> dict[str, Any]:
    """The filter as one query (``match_all`` when it constrains nothing), for
    routes that count or aggregate over the matching items."""
    clauses = item_filter_clauses(f, embedding=embedding, image_join=image_join)
    return {'bool': {'filter': clauses}} if clauses else {'match_all': {}}


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
    'MAX_SELECTION_IDS',
    'QUERY_IDS_MAX',
    'SAM3_ORIGIN_FIELD',
    'ItemFilter',
    'Origin',
    'ReviewStatus',
    'item_filter_clauses',
    'item_filter_query',
    'visibility_clauses',
]
