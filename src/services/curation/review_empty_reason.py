"""A real-state reason for a zero-result ``GET /review/{tab}``.

Split out of :mod:`review_queries` (which owns tab *selection*) to keep
that module under the 700-LOC pre-commit ceiling -- this module owns
*why a tab came back empty*, computed from live index state rather than
a canned string, so the labeler can tell an operator what to actually do
("run a probe") instead of a bare "Queue empty."
"""

from __future__ import annotations

from typing import Any

from src.config import IndexRole, get_curation_config, index_name
from src.services.curation.embedding_state import not_embedded_clause
from src.services.curation.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE


# Tabs whose own selection requires a probe-scored item
# (review_queries.build_tab_query's 'uncertainty' / 'model_disagreements'
# branches both add an `exists` clause on one of these fields) -- an
# empty result on one of these almost always means no probe pass has run
# yet, not that the pool has nothing uncertain.
_PROBE_GATED_TABS: dict[str, str] = {
    'uncertainty': 'probe_pred_entropy',
    'model_disagreements': 'probe_pred_class',
}


def _items_index() -> str:
    return index_name(get_curation_config(), IndexRole.ITEMS)


async def _field_has_any_value(opensearch: Any, field: str) -> bool:
    resp = await opensearch.count(
        index=_items_index(), body={'query': {'exists': {'field': field}}}
    )
    return int(resp.get('count', 0)) > 0


async def _count_unembedded(opensearch: Any) -> int:
    resp = await opensearch.count(index=_items_index(), body={'query': not_embedded_clause()})
    return int(resp.get('count', 0))


async def _imported_labels_exist(opensearch: Any) -> bool:
    resp = await opensearch.count(
        index=_items_index(), body={'query': {'term': {'class_source': LABEL_IMPORT_CLASS_SOURCE}}}
    )
    return int(resp.get('count', 0)) > 0


REGION_PROFILE_OFF_REASON = (
    'the region profile is off for this project: activate one to review regions'
)


def region_queue_is_off(tab: str) -> bool:
    """The ``regions`` tab has nothing to serve while no region profile is
    active: rows written under an earlier profile are stale, not a queue."""
    from src.services.detection.profile_registry import get_active_region_profile

    return tab == 'regions' and get_active_region_profile() is None


async def compute_empty_reason(tab: str, filters: Any, opensearch: Any) -> str:
    """A real-state reason for a zero-result ``GET /review/{tab}``:

    - a probe-gated tab (``uncertainty`` / ``model_disagreements``) with
      no probe-scored item at all -> "no probe predictions — run a probe";
    - the caller asked for a ``min_mistakenness`` floor but no item has
      ever been scored -> "item scores never computed";
    - ``new_class_proposals`` with nothing pending -> "no unclassified
      proposals";
    - ``imported`` with an import filter set -> "no imported labels match
      these filters", with no imported label at all -> "no imported labels:
      import a dataset first";
    - otherwise -> "no items match".
    """
    probe_field = _PROBE_GATED_TABS.get(tab)
    if probe_field is not None and not await _field_has_any_value(opensearch, probe_field):
        return 'no probe predictions — run a probe'
    if getattr(filters, 'min_mistakenness', None) is not None and not await _field_has_any_value(
        opensearch, 'mistakenness_score'
    ):
        return 'item scores never computed'
    if tab == 'new_class_proposals':
        return 'no unclassified proposals'
    if tab == 'imported':
        if getattr(filters, 'import_id', None) or getattr(filters, 'dataset_split', None):
            return 'no imported labels match these filters'
        if not await _imported_labels_exist(opensearch):
            return 'no imported labels: import a dataset first'
    if tab == 'all':
        unembedded = await _count_unembedded(opensearch)
        if unembedded:
            return f'{unembedded} items have no embedding, so this queue skips them: embed them'
    return 'no items match'


async def review_tabs_empty_state(opensearch: Any) -> dict[str, bool]:
    """The underlying-state flags :func:`compute_empty_reason` reads,
    served once on ``GET /review/tabs`` so a client can annotate ANY
    tab's zero-result state without one ``count`` round-trip per tab."""
    return {
        'has_probe_predictions': await _field_has_any_value(opensearch, 'probe_pred_entropy'),
        'has_item_scores': await _field_has_any_value(opensearch, 'mistakenness_score'),
        'has_imported_labels': await _imported_labels_exist(opensearch),
    }


__all__ = [
    'REGION_PROFILE_OFF_REASON',
    'compute_empty_reason',
    'region_queue_is_off',
    'review_tabs_empty_state',
]
