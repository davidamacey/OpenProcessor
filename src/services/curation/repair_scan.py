"""The paged candidate scan the item repair jobs share: search ``index`` for the repair's
candidate query in ``crop_id`` order (``search_after``) and plan each hit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


async def scan_plans[PlanT](
    client: Any,
    *,
    index: str,
    query: dict[str, Any],
    plan_one: Callable[[str, dict[str, Any]], PlanT | None],
    page_size: int,
) -> list[PlanT]:
    """``plan_one(crop_id, source)`` for every hit of ``query``; ``None`` means not a
    candidate. Embedding vectors are not fetched."""
    plans: list[PlanT] = []
    cursor: list[Any] | None = None
    while True:
        body: dict[str, Any] = {
            'size': page_size,
            'query': query,
            'sort': [{'crop_id': 'asc'}],
            '_source': {'excludes': ['*embedding*']},
        }
        if cursor is not None:
            body['search_after'] = cursor
        resp = await client.search(index=index, body=body)
        hits = (resp.get('hits') or {}).get('hits') or []
        if not hits:
            break
        cursor = hits[-1].get('sort')
        for h in hits:
            plan = plan_one(h['_id'], h.get('_source') or {})
            if plan is not None:
                plans.append(plan)
        if len(hits) < page_size or cursor is None:
            break
    return plans
