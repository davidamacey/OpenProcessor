"""What a near-duplicate group member shows: its picture, class, holdout state and pin.

A member id is an image id at the image level and an item (crop) id at the item level. Pictures
are served URLs derived from the id (no lookup); the class and the holdout state come from one
query of the items index per group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.curation_accessors import items_index
from src.config.project_context import project_api_base
from src.schemas.curation import near_duplicates as wire


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.db.repos import near_dup as repo


def thumbnail_url(level: wire.Level, member_id: str) -> str:
    """The served picture: an image thumbnail (``GET /images/{id}/thumbnail``) or a crop thumbnail."""
    kind = 'images' if level == 'image' else 'crops'
    return f'{project_api_base()}/{kind}/{member_id}/thumbnail'


async def item_facts(
    opensearch: Any, level: wire.Level, member_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """``member_id -> {image_id, class_name, in_holdout}`` from one items-index query."""
    if not member_ids:
        return {}
    if level == 'item':
        resp = await opensearch.search(
            index=items_index(),
            body={
                'size': len(member_ids),
                'query': {'ids': {'values': list(member_ids)}},
                '_source': ['image_id', 'class_name', 'test_holdout'],
            },
        )
        return {
            h['_id']: {
                'image_id': h['_source'].get('image_id'),
                'class_name': h['_source'].get('class_name'),
                'in_holdout': bool(h['_source'].get('test_holdout')),
            }
            for h in resp['hits']['hits']
        }
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 0,
            'query': {
                'bool': {
                    'filter': [
                        {'terms': {'image_id': list(member_ids)}},
                        {'term': {'test_holdout': True}},
                    ]
                }
            },
            'aggs': {'held': {'terms': {'field': 'image_id', 'size': len(member_ids)}}},
        },
    )
    held = {b['key'] for b in resp['aggregations']['held']['buckets']}
    return {m: {'image_id': m, 'class_name': None, 'in_holdout': m in held} for m in member_ids}


def pins_to(
    representative_id: str, decisions: Sequence[repo.DecisionRow]
) -> dict[str, wire.NearDupPin]:
    """``member_id -> pin`` for every decision between a member and the representative."""
    pins: dict[str, wire.NearDupPin] = {}
    for d in decisions:
        other = d.b_id if d.a_id == representative_id else d.a_id
        if representative_id in (d.a_id, d.b_id):
            pins[other] = wire.NearDupPin(id=str(d.id), kind=d.decision)
    return pins


def member_view(
    level: wire.Level,
    row: tuple[str, bool, float],
    facts: dict[str, dict[str, Any]],
    pins: dict[str, wire.NearDupPin],
) -> wire.NearDupMemberView:
    member_id, is_representative, cosine = row
    fact = facts.get(member_id, {})
    return wire.NearDupMemberView(
        member_id=member_id,
        is_representative=is_representative,
        cosine_to_rep=cosine,
        thumbnail_url=thumbnail_url(level, member_id),
        image_id=fact.get('image_id'),
        crop_id=member_id if level == 'item' else None,
        class_name=fact.get('class_name'),
        in_holdout=bool(fact.get('in_holdout', False)),
        decision=pins.get(member_id),
    )
