"""``GET /stats/classes``: per-class total/validated breakdown joined to the registry names.

``trainable`` = validated minus frozen test-holdout minus human-excluded crops, so every page
agrees on the number (see :mod:`src.services.curation.class_counts`). ``trainable_gap`` is the
shortfall against :func:`dataset_thresholds`'s per-class minimum, floored at 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.class_counts import fetch_class_counts
from src.services.curation.dataset_thresholds import adequacy, aug_target, dataset_thresholds
from src.services.curation.stats_dataset_query import StatsQueryError, filter_query
from src.services.embedding_spaces.readers import item_write_fields


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.services.classes.registry import ClassRegistry
    from src.services.curation.item_filter import ItemFilter


class ClassRegistryUnavailableError(Exception):
    """The class registry could not be read."""


async def class_stats(
    opensearch: Any, item_filter: ItemFilter, registry: Callable[[], ClassRegistry]
) -> dict[str, Any]:
    """The raw ``by_class`` aggregation plus a flattened ``classes`` list and the ``thresholds``
    behind it. Raises ``InvalidItemFilterError`` for a filter that cannot be built,
    :class:`StatsQueryError` when the search fails and :class:`ClassRegistryUnavailableError`
    when the registry fails."""
    query = (
        None
        if item_filter.is_empty()
        else (await filter_query(opensearch, item_filter, await item_write_fields()))[0]
    )
    try:
        counts, aggs = await fetch_class_counts(opensearch, query=query, by_source=True)
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
                }
            )
    except Exception as exc:
        raise ClassRegistryUnavailableError from exc

    return {**aggs, 'classes': classes, 'thresholds': threshold_values}
