"""Query helpers for ``GET /crops``: the ``sort`` whitelist and the
confidence band. Kept out of the router so it stays a thin HTTP layer."""

from __future__ import annotations

from typing import Any


# Sortable item fields -> the ``unmapped_type`` OpenSearch needs when a
# shard has no document carrying the field yet.
CROP_SORT_FIELDS: dict[str, str] = {
    'updated_at': 'date',
    'created_at': 'date',
    'confidence': 'float',
    'classifier_raw_confidence': 'float',
    'crop_rank_in_image': 'integer',
    'crop_area_norm': 'float',
    'blur_lap_ratio': 'float',
    'cluster_distance': 'float',
    'mistakenness_score': 'float',
    'uniqueness_score': 'float',
}

DEFAULT_CROP_SORT = 'updated_at:desc'


def parse_crop_sort(sort: str | None) -> list[dict[str, Any]]:
    """``'<field>[:asc|desc]'`` -> an OpenSearch sort clause.

    Raises ``ValueError`` (the router maps it to 400) for an unknown field
    or direction, listing the valid choices. Missing values sort last in
    either direction.
    """
    spec = (sort or DEFAULT_CROP_SORT).strip()
    field, _, direction = spec.partition(':')
    field = field.strip()
    direction = (direction.strip() or 'desc').lower()
    if field not in CROP_SORT_FIELDS:
        msg = f'unknown sort field {field!r}; choose from {sorted(CROP_SORT_FIELDS)}'
        raise ValueError(msg)
    if direction not in ('asc', 'desc'):
        msg = f"unknown sort direction {direction!r}; use 'asc' or 'desc'"
        raise ValueError(msg)
    return [
        {
            field: {
                'order': direction,
                'missing': '_last',
                'unmapped_type': CROP_SORT_FIELDS[field],
            }
        },
        # F-7: stable tiebreaker. crop_id is a mapped keyword field equal to
        # _id -- sort on it directly rather than _id (which uses fielddata,
        # disabled on these indexes) so ties on the primary sort key don't
        # produce duplicate/skipped rows across pages.
        {'crop_id': {'order': 'asc'}},
    ]


def confidence_band(conf_min: float | None, conf_max: float | None) -> dict[str, Any] | None:
    """Inclusive ``confidence`` range clause, or None when neither bound is set."""
    if conf_min is None and conf_max is None:
        return None
    if conf_min is not None and conf_max is not None and conf_min > conf_max:
        msg = f'conf_min ({conf_min}) is greater than conf_max ({conf_max})'
        raise ValueError(msg)
    rng: dict[str, float] = {}
    if conf_min is not None:
        rng['gte'] = conf_min
    if conf_max is not None:
        rng['lte'] = conf_max
    return {'range': {'confidence': rng}}


def crops_page(
    *,
    total: int,
    page: int,
    page_size: int,
    crops: list[dict[str, Any]],
    method: str | None = None,
    version: str | None = None,
    n_pool: int | None = None,
) -> dict[str, Any]:
    """``CropsPageResponse``-shaped envelope around serialized items."""
    return {
        'total': total,
        'page': page,
        'page_size': page_size,
        'crops': crops,
        'method': method,
        'version': version,
        'n_pool': n_pool,
    }


__all__ = [
    'CROP_SORT_FIELDS',
    'DEFAULT_CROP_SORT',
    'confidence_band',
    'crops_page',
    'parse_crop_sort',
]
