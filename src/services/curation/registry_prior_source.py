"""Build the registry prior from served project state (never from the client)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.curation import items_index
from src.services.labeling.registry_prior import RegistryPrior, rank_registry_prior


if TYPE_CHECKING:
    from collections.abc import Collection

_PENDING_BUCKETS = 200


class RegistryPriorUnavailableError(RuntimeError):
    """The pack asks for a registry prior but the counts could not be read.

    Fail closed: labeling without the prior the pack asked for would silently
    change the prompt the run is stamped with.
    """


def _buckets(aggs: dict[str, Any], key: str) -> dict[str, int]:
    node = aggs.get(key, {}).get('by_name', {})
    return {str(b['key']): int(b['doc_count']) for b in node.get('buckets', [])}


async def load_registry_prior(
    opensearch: Any, *, top_k: int, registry_names: Collection[str]
) -> RegistryPrior | None:
    """The prior for ``top_k`` > 0 (``None`` when off or nothing to rank)."""
    if top_k <= 0 or not registry_names:
        return None
    body = {
        'size': 0,
        'query': {'bool': {'must_not': [{'term': {'test_holdout': True}}]}},
        'aggs': {
            'validated': {
                'filter': {'term': {'class_validated': True}},
                'aggs': {
                    'by_name': {'terms': {'field': 'class_name', 'size': len(registry_names)}}
                },
            },
            'pending': {
                'filter': {'term': {'class_source': 'vlm_new_class_pending'}},
                'aggs': {
                    'by_name': {'terms': {'field': 'vlm_proposed_class', 'size': _PENDING_BUCKETS}}
                },
            },
        },
    }
    try:
        resp = await opensearch.search(index=items_index(), body=body)
        aggs = resp['aggregations']
        validated = _buckets(aggs, 'validated')
        pending = _buckets(aggs, 'pending')
    except Exception as exc:
        raise RegistryPriorUnavailableError(f'registry prior counts unavailable: {exc}') from exc
    return rank_registry_prior(registry_names, validated, pending, top_k=top_k)


async def prior_for_pack(
    opensearch: Any, pack: Any, registry_names: Collection[str]
) -> RegistryPrior | None:
    """The prior the pack asks for (``None`` when its ``registry_prior_top_k`` is 0)."""
    return await load_registry_prior(
        opensearch, top_k=pack.registry_prior_top_k, registry_names=registry_names
    )


async def prior_or_error(
    opensearch: Any, pack: Any, registry_names: Collection[str]
) -> tuple[RegistryPrior | None, str | None]:
    """:func:`prior_for_pack`, with an unreadable-counts failure as its message."""
    try:
        return await prior_for_pack(opensearch, pack, registry_names), None
    except RegistryPriorUnavailableError as exc:
        return None, str(exc)
