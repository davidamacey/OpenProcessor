"""``GET /stats/classes``: per-class total/validated breakdown joined to the registry names.

``trainable`` = validated minus frozen test-holdout minus human-excluded crops, so every page
agrees on the number (see :mod:`src.services.curation.class_counts`). ``trainable_gap`` is the
shortfall against :func:`dataset_thresholds`'s per-class minimum, floored at 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.class_counts import fetch_class_counts
from src.services.curation.dataset_thresholds import adequacy, aug_target, dataset_thresholds
from src.services.curation.holdout_freeze import holdout_stats
from src.services.curation.stats_dataset_query import StatsQueryError, filter_query
from src.services.embedding_spaces.readers import item_write_fields


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.schemas.curation.views import SortOption
    from src.services.classes.registry import ClassRegistry
    from src.services.curation.item_filter import ItemFilter


# sort field -> label; the field is a key of the served class row. Missing never happens: every
# row carries every field, and ties end on ``class_id`` ascending so the order is total.
CLASS_SORT_LABELS: dict[str, str] = {
    'class_name': 'Class',
    'class_id': 'ID',
    'count': 'Total (labelled)',
    'validated_count': 'Validated',
    'trainable': 'Trainable',
    'aug_target': 'Aug target',
    'trainable_gap': 'Gap',
    'test_count': 'Test (held out)',
}
DEFAULT_CLASS_SORT = 'class_id:asc'
# A bare field sorts names and ids ascending and every count largest first.
ASCENDING_BY_DEFAULT = frozenset({'class_name', 'class_id'})
CLASS_SORT_OPTIONS: list[SortOption] = [
    {'id': f'{field}:{direction}', 'label': f'{label} ({direction})'}
    for field, label in CLASS_SORT_LABELS.items()
    for direction in ('asc', 'desc')
]


def parse_class_sort(sort: str | None) -> tuple[str, bool]:
    """``'<field>[:asc|desc]'`` -> ``(field, descending)``; a bare field takes its default direction.

    Raises ``ValueError`` (the router maps it to 400) for an unknown field or direction."""
    field, _, direction = (sort or DEFAULT_CLASS_SORT).strip().partition(':')
    field = field.strip()
    direction = (direction.strip() or ('asc' if field in ASCENDING_BY_DEFAULT else 'desc')).lower()
    if field not in CLASS_SORT_LABELS:
        msg = f'unknown sort field {field!r}; choose from {sorted(CLASS_SORT_LABELS)}'
        raise ValueError(msg)
    if direction not in ('asc', 'desc'):
        msg = f"unknown sort direction {direction!r}; use 'asc' or 'desc'"
        raise ValueError(msg)
    return field, direction == 'desc'


def sort_class_rows(rows: list[dict[str, Any]], field: str, *, descending: bool) -> None:
    """Order ``rows`` in place by ``field`` (names compare case-insensitively), ties by class id."""
    rows.sort(key=lambda r: r['class_id'])
    rows.sort(
        key=lambda r: r[field].casefold() if field == 'class_name' else r[field],
        reverse=descending,
    )


class ClassRegistryUnavailableError(Exception):
    """The class registry could not be read."""


async def class_stats(
    opensearch: Any,
    item_filter: ItemFilter,
    registry: Callable[[], ClassRegistry],
    sort: str | None = None,
) -> dict[str, Any]:
    """The raw ``by_class`` aggregation plus a flattened ``classes`` list and the ``thresholds``
    behind it. Raises ``InvalidItemFilterError`` for a filter that cannot be built,
    :class:`StatsQueryError` when the search fails and :class:`ClassRegistryUnavailableError`
    when the registry fails, and ``ValueError`` for an unknown ``sort``."""
    sort_field, descending = parse_class_sort(sort)
    query = (
        None
        if item_filter.is_empty()
        else (await filter_query(opensearch, item_filter, await item_write_fields()))[0]
    )
    try:
        counts, aggs = await fetch_class_counts(opensearch, query=query, by_source=True)
        held_out = {b['key']: b for b in (await holdout_stats(opensearch))['by_class']}
    except Exception as exc:
        raise StatsQueryError from exc

    threshold_values = dataset_thresholds()
    hard_min = int(threshold_values.get('block_below', 0))

    classes: list[dict[str, Any]] = []
    try:
        reg = await registry().snapshot()
        for c in reg.classes:
            if getattr(c, 'deprecated', False):
                continue
            n_valid = counts.validated.get(c.class_id, 0)
            n_trainable = counts.trainable(c.class_id)
            target = aug_target(n_valid)
            test = held_out.get(c.class_id)
            classes.append(
                {
                    'class_id': c.class_id,
                    'class_name': c.class_name,
                    'count': counts.count.get(c.class_id, 0),
                    'validated_count': n_valid,
                    'trainable': n_trainable,
                    'trainable_gap': max(0, hard_min - n_trainable),
                    'adequacy': adequacy(n_valid),
                    'aug_target': target,
                    'aug_gap': target - n_valid,
                    'test_count': int(test['doc_count']) if test else 0,
                    'test_deficient': bool(test['deficient']) if test else False,
                }
            )
    except Exception as exc:
        raise ClassRegistryUnavailableError from exc

    sort_class_rows(classes, sort_field, descending=descending)
    return {
        **aggs,
        'classes': classes,
        'thresholds': threshold_values,
        'sort_applied': f'{sort_field}:{"desc" if descending else "asc"}',
        'sort_options': CLASS_SORT_OPTIONS,
    }
