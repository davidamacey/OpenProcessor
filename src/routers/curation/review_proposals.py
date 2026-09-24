"""``GET /curation/review/new_class_proposals/summary`` — the proposed
new-class names over exactly the new-class queue, with junk terms flagged.

Split out of ``review.py`` (LOC ceiling). Selection and term rules live in
:mod:`src.services.curation.new_class_terms`, shared with the queue and
with ``POST /review/new_class_proposals/resolve``.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Query

from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    _ensure_indexes,
    get_class_registry,
    router,
)
from src.services.curation.new_class_terms import (
    PROPOSED_NAME_FIELD,
    classify_term,
    load_term_rules,
    proposal_query,
)


def _sample_ids(bucket: dict[str, Any]) -> list[Any]:
    hits = ((bucket.get('samples') or {}).get('hits') or {}).get('hits') or []
    return [(h.get('_source') or {}).get('crop_id') or h.get('_id') for h in hits]


@router.get('/review/new_class_proposals/summary')
async def review_new_class_summary(
    opensearch: OpenSearchDep,
    size: Annotated[int, Query(ge=1, le=1000)] = 100,
    samples: Annotated[int, Query(ge=0, le=20)] = 5,
) -> dict[str, Any]:
    """The proposed new-class names across the ``new_class_proposals`` queue.

    ``{total_pending, without_term, top_terms, flagged_terms, term_rules}``:
    ``total_pending`` equals ``GET /review/new_class_proposals``'s ``total``
    (same selection); ``without_term`` counts queue items with no proposed
    name (e.g. a human flag). Each term is ``{label, count,
    sample_crop_ids, flag, class_id}``, most common first; ``count`` equals
    what a resolve for that label matches. ``top_terms`` holds terms worth
    creating (``flag`` null); ``flagged_terms`` holds the rest, ``flag``
    one of ``existing_class`` (``class_id`` set: map to it),
    ``generic_parent`` or ``non_object``. ``term_rules`` serves the rule.
    """
    await _ensure_indexes(opensearch)
    terms: dict[str, Any] = {
        'terms': {'field': PROPOSED_NAME_FIELD, 'size': size, 'min_doc_count': 1}
    }
    if samples:
        terms['aggs'] = {'samples': {'top_hits': {'size': samples, '_source': ['crop_id']}}}
    body = {
        'size': 0,
        'query': proposal_query(),
        'aggs': {
            'proposed': terms,
            'without_term': {
                'filter': {'bool': {'must_not': [{'exists': {'field': PROPOSED_NAME_FIELD}}]}}
            },
        },
        'track_total_hits': True,
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    rules = load_term_rules(get_class_registry())
    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    aggs = resp.get('aggregations') or {}
    top_terms: list[dict[str, Any]] = []
    flagged_terms: list[dict[str, Any]] = []
    for b in (aggs.get('proposed') or {}).get('buckets') or []:
        label = str(b.get('key', ''))
        flag, class_id = classify_term(label, rules)
        entry = {
            'label': label,
            'count': int(b.get('doc_count', 0)),
            'sample_crop_ids': _sample_ids(b),
            'flag': flag,
            'class_id': class_id,
        }
        (flagged_terms if flag else top_terms).append(entry)
    return {
        'total_pending': int(total),
        'without_term': int((aggs.get('without_term') or {}).get('doc_count', 0)),
        'top_terms': top_terms,
        'flagged_terms': flagged_terms,
        'term_rules': rules.to_wire(),
    }
