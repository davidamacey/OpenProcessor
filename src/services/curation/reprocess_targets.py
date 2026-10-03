"""Resolve a reprocess request's targets to documents (W10.13).

Three target forms: explicit ``crop_ids``, explicit ``image_ids`` or a
``filter``. This module validates the shape (exactly one form, a filter
that actually selects something), turns a filter into one OpenSearch
query and enumerates the matching items / images. It never writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.region_fields import RegionFields, get_region_fields
from src.services.curation.item_filter import item_filter_clauses
from src.services.curation.region_requeue import NONE_BUCKET, box_values_filter
from src.services.curation.reprocess_models import MAX_TARGET_IDS, ReprocessFilter, ReprocessTargets


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

_PAGE = 500
_ID_CHUNK = 1000


class ReprocessTargetsError(ValueError):
    """The request's targets are malformed (maps to 422
    ``reprocess_targets_invalid``)."""


def validate_targets(targets: ReprocessTargets) -> str:
    """The one target form in use: ``image_ids``, ``crop_ids`` or ``filter``.

    Raises :class:`ReprocessTargetsError` for none, several, an empty id
    list, more than :data:`MAX_TARGET_IDS` ids, or a filter with no field set
    (an empty filter would mean "every item").
    """
    given = [
        name
        for name, value in (
            ('image_ids', targets.image_ids),
            ('crop_ids', targets.crop_ids),
            ('filter', targets.filter),
        )
        if value is not None
    ]
    if len(given) != 1:
        raise ReprocessTargetsError(
            'exactly one of image_ids, crop_ids or filter is required '
            f'(got {", ".join(given) or "none"})'
        )
    kind = given[0]
    if kind in ('image_ids', 'crop_ids'):
        ids = getattr(targets, kind)
        if not ids:
            raise ReprocessTargetsError(f'{kind} is empty')
        if len(ids) > MAX_TARGET_IDS:
            raise ReprocessTargetsError(f'{kind} has {len(ids)} ids; the limit is {MAX_TARGET_IDS}')
    elif targets.filter is not None and targets.filter.is_empty():
        raise ReprocessTargetsError('the filter selects nothing specific; set at least one field')
    if kind != 'filter' and (targets.limit is not None or targets.sample is not None):
        raise ReprocessTargetsError('limit and sample apply to a filter, not to explicit ids')
    if targets.sample is not None and targets.limit is None:
        raise ReprocessTargetsError('sample needs a limit')
    if (
        targets.limit is not None
        and targets.filter is not None
        and has_image_selector(targets.filter)
    ):
        raise ReprocessTargetsError('limit applies to item filters, not to image-level selectors')
    if targets.filter is not None:
        try:
            item_filter_clauses(targets.filter)
        except ValueError as exc:
            raise ReprocessTargetsError(str(exc)) from exc
    return kind


def selector_clauses(f: ReprocessFilter, F: RegionFields | None = None) -> list[dict[str, Any]]:
    """Clauses every scope shares: the profile stamp selectors plus the item
    filter."""
    F = F or get_region_fields()
    out: list[dict[str, Any]] = []
    if f.profile_not is not None and f.profile_revision_below is not None:
        out.append(
            {
                'bool': {
                    'should': [
                        {'bool': {'must_not': [{'term': {F.profile: f.profile_not}}]}},
                        {
                            'bool': {
                                'filter': [
                                    {'term': {F.profile: f.profile_not}},
                                    {
                                        'range': {
                                            F.profile_revision: {'lt': f.profile_revision_below}
                                        }
                                    },
                                ]
                            }
                        },
                    ],
                    'minimum_should_match': 1,
                }
            }
        )
    elif f.profile_not is not None:
        out.append({'bool': {'must_not': [{'term': {F.profile: f.profile_not}}]}})
    elif f.profile_revision_below is not None:
        out.append({'range': {F.profile_revision: {'lt': f.profile_revision_below}}})
    return [*out, *item_filter_clauses(f)]


def has_image_selector(f: ReprocessFilter) -> bool:
    """The filter selects images (not items): see :class:`ReprocessFilter`."""
    return f.all_images or bool(f.open_vocab_status)


def image_filter_query(f: ReprocessFilter) -> dict[str, Any]:
    """The image-level selectors as one images query."""
    if f.open_vocab_status:
        return {'terms': {'open_vocab_status': list(f.open_vocab_status)}}
    return {'match_all': {}}


def has_profile_selector(f: ReprocessFilter) -> bool:
    return f.profile_not is not None or f.profile_revision_below is not None


def item_filter_query(f: ReprocessFilter, F: RegionFields | None = None) -> dict[str, Any]:
    """The whole filter as one items query (the non-region scopes; the
    region scope builds a :class:`~...region_requeue.RequeueSelection`
    per status instead)."""
    F = F or get_region_fields()
    filt = selector_clauses(f, F)
    must_not: list[dict[str, Any]] = []
    if f.region_status:
        filt.append({'terms': {F.status: list(f.region_status)}})
    if f.missing_status:
        must_not.append({'exists': {'field': F.status}})
    if (
        box_filter := box_values_filter(F, detectors=tuple(f.detector), reasons=tuple(f.reason))
    ) is not None:
        filt.append(box_filter)
    if f.missing_provenance:
        must_not.append({'exists': {'field': F.detector_chain}})
    return {'bool': {'filter': filt, 'must_not': must_not}}


async def scan_items(
    opensearch: AsyncOpenSearch,
    query: dict[str, Any],
    *,
    index: str,
    includes: list[str],
    max_docs: int = 0,
    id_field: str = 'crop_id',
) -> list[tuple[str, dict[str, Any]]]:
    """Every ``(id, _source)`` the query matches (``search_after`` pages
    sorted by ``id_field``: ``crop_id`` for items, ``image_id`` for images),
    ``_source`` limited to ``includes``."""
    out: list[tuple[str, dict[str, Any]]] = []
    cursor: list[Any] | None = None
    while True:
        body: dict[str, Any] = {
            'size': _PAGE,
            '_source': includes,
            'query': query,
            'sort': [{id_field: 'asc'}],
        }
        if cursor is not None:
            body['search_after'] = cursor
        resp = await opensearch.search(index=index, body=body)
        hits = (resp.get('hits') or {}).get('hits') or []
        if not hits:
            break
        out.extend((h['_id'], h.get('_source') or {}) for h in hits)
        cursor = hits[-1].get('sort')
        if cursor is None or len(hits) < _PAGE or (max_docs and len(out) >= max_docs):
            break
    return out[:max_docs] if max_docs else out


async def items_by_terms(
    opensearch: AsyncOpenSearch,
    field: str,
    values: list[str],
    *,
    index: str,
    includes: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Items whose ``field`` (``crop_id`` or ``image_id``) is one of
    ``values``."""
    out: list[tuple[str, dict[str, Any]]] = []
    for start in range(0, len(values), _ID_CHUNK):
        chunk = values[start : start + _ID_CHUNK]
        out.extend(
            await scan_items(
                opensearch,
                {'bool': {'filter': [{'terms': {field: chunk}}]}},
                index=index,
                includes=includes,
            )
        )
    return out


async def existing_images(
    opensearch: AsyncOpenSearch, image_ids: list[str], *, index: str
) -> dict[str, dict[str, Any]]:
    """``{image_id: images-doc source}`` for the ids that exist."""
    found: dict[str, dict[str, Any]] = {}
    for start in range(0, len(image_ids), _ID_CHUNK):
        chunk = image_ids[start : start + _ID_CHUNK]
        resp = await opensearch.search(
            index=index,
            body={
                'size': len(chunk),
                '_source': ['image_id', 'image_path', 'dataset_split'],
                'query': {'terms': {'image_id': chunk}},
            },
        )
        for hit in (resp.get('hits') or {}).get('hits') or []:
            src = hit.get('_source') or {}
            found[src.get('image_id') or hit['_id']] = src
    return found


__all__ = [
    'NONE_BUCKET',
    'ReprocessTargetsError',
    'existing_images',
    'has_image_selector',
    'has_profile_selector',
    'image_filter_query',
    'item_filter_query',
    'items_by_terms',
    'scan_items',
    'selector_clauses',
    'validate_targets',
]
