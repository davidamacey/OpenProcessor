"""The classes x cohorts matrix behind ``GET /training_cohorts/summary``.

One search per cohort, each with a terms aggregation over the items it matches: the counts for every
class come back together, so the number of searches is the number of cohorts, never classes x
cohorts. The queries are the ones that serve a cohort's rows (the router builds them); this module
shapes the answers into rows.

The "detector proposals not decided" cohort is the review queue ``to_label`` (the item has no class
yet, nobody has decided it). It is counted by the detector's own class name, because the item has
no class to count it under: a name that matches an active registry class lands in that class's row,
any other name gets a row of its own (``kind: detector``), so a label the registry lacks is visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.training_cohorts import CohortSpec
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from src.clients.curation_opensearch import RegistryClassEntry

MAX_BUCKETS = 5000
"""Classes (or detector names) one cohort's aggregation reads."""

UNDECIDED_ID = 'detector_proposals'
UNDECIDED = CohortSpec(
    'Detector proposals not decided',
    'The detector named a class for the box and nobody has given the item a class yet',
    '/crops',
    {'review_tab': 'to_label'},
)


def agg_body(query: dict[str, Any], field: str) -> dict[str, Any]:
    """A count-only search of ``query`` with its hits grouped by ``field``."""
    return {
        'size': 0,
        'track_total_hits': True,
        'query': query,
        'aggs': {'groups': {'terms': {'field': field, 'size': MAX_BUCKETS}}},
    }


def _total(response: Mapping[str, Any]) -> int:
    return int(((response.get('hits') or {}).get('total') or {}).get('value', 0))


def _buckets(response: Mapping[str, Any]) -> dict[Any, int]:
    groups = (response.get('aggregations') or {}).get('groups') or {}
    return {b['key']: int(b['doc_count']) for b in groups.get('buckets') or []}


def active_classes(classes: Iterable[RegistryClassEntry]) -> list[RegistryClassEntry]:
    """The classes a row is served for: active, and not merged into another."""
    return [c for c in classes if not c.deprecated and getattr(c, 'merged_into', None) is None]


def build_summary(
    cohorts: Sequence[dict[str, Any]],
    class_responses: Mapping[str, Mapping[str, Any]],
    undecided_response: Mapping[str, Any],
    classes: Iterable[RegistryClassEntry],
) -> dict[str, Any]:
    """The summary body.

    ``cohorts`` are the catalog entries (``cohort_catalog``), ``class_responses`` the answer of
    each one's search by ``class_id`` (keyed by cohort id), ``undecided_response`` the answer of
    the undecided cohort's search by ``detector_class_name``."""
    rows: dict[str, dict[str, Any]] = {}
    by_name: dict[str, str] = {}
    cohort_ids = [c['id'] for c in cohorts]
    undecided_total = _total(undecided_response)
    if undecided_total:
        cohort_ids.append(UNDECIDED_ID)

    def new_row(class_id: int | None, name: str, kind: str) -> dict[str, Any]:
        return {
            'class_id': class_id,
            'class_name': name,
            'kind': kind,
            'counts': dict.fromkeys(cohort_ids, 0),
        }

    for c in active_classes(classes):
        rows[f'c{c.class_id}'] = new_row(c.class_id, c.class_name, 'class')
        by_name[normalize_class_name(c.class_name)] = f'c{c.class_id}'
    for cohort in cohorts:
        for class_id, count in _buckets(class_responses[cohort['id']]).items():
            row = rows.get(f'c{class_id}')
            if row is not None:
                row['counts'][cohort['id']] = count
    for name, count in _buckets(undecided_response).items():
        slug = normalize_class_name(str(name))
        if not slug:
            continue
        key = by_name.get(slug)
        if key is None:
            key = f'd{slug}'
            rows.setdefault(key, new_row(None, slug, 'detector'))
        rows[key]['counts'][UNDECIDED_ID] = count
    served = [{**c, 'total': _total(class_responses[c['id']])} for c in cohorts]
    if undecided_total:
        served.append(
            {
                'id': UNDECIDED_ID,
                'label': UNDECIDED.label,
                'description': UNDECIDED.description,
                'cutoffs': {},
                'endpoint': UNDECIDED.endpoint,
                'params': dict(UNDECIDED.params),
                'row_kind': UNDECIDED.row_kind,
                'total': undecided_total,
            }
        )
    ordered = sorted(
        rows.values(), key=lambda r: (r['kind'] != 'class', r['class_id'] or 0, r['class_name'])
    )
    return {'cohorts': served, 'rows': ordered}


__all__ = ['MAX_BUCKETS', 'UNDECIDED', 'UNDECIDED_ID', 'agg_body', 'build_summary']
