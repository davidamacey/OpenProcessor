"""The paged candidate scan the item repair jobs share: search ``index`` for the repair's
candidate query in ``crop_id`` order (``search_after``) and plan each hit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from opensearchpy.exceptions import OpenSearchException

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.services.embedding_spaces.readers import vector_source_excludes


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class _HasCropId(Protocol):
    crop_id: str


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


def planner[PlanT](
    candidate_query: Callable[..., dict[str, Any]],
    plan_one: Callable[[str, dict[str, Any]], PlanT | None],
) -> Callable[..., Awaitable[list[PlanT]]]:
    """The repair's ``plan_repairs(client, *, index, id_prefix, page_size)``: scan ``index``
    for ``candidate_query(id_prefix=...)`` and plan each hit with ``plan_one``."""

    async def plan_repairs(
        client: Any,
        *,
        index: str,
        id_prefix: str | None = None,
        page_size: int = 500,
    ) -> list[PlanT]:
        """Scan ``index`` for candidates (paged by ``crop_id``) and plan each."""
        return await scan_plans(
            client,
            index=index,
            query=candidate_query(id_prefix=id_prefix),
            plan_one=plan_one,
            page_size=page_size,
        )

    return plan_repairs


class StaleRepairError(Exception):
    """Raised by a merger when the doc no longer matches its plan; the item is skipped."""


async def apply_occ_repairs[PlanT: _HasCropId](
    client: Any,
    plans: list[PlanT],
    *,
    index: str,
    writer_id: str,
    merger_for: Callable[[PlanT], Callable[[dict[str, Any]], dict[str, Any]]],
    counts: dict[str, int],
    is_applicable: Callable[[PlanT], bool] | None = None,
) -> dict[str, int]:
    """Write each plan under OCC with ``merger_for(plan)`` and tally into ``counts``
    (``repaired`` / ``skipped_changed`` / ``errors``, plus ``not_applicable`` when
    ``is_applicable`` rejects a plan). A merger that raises :class:`StaleRepairError`
    or an OCC final conflict counts as ``skipped_changed``."""
    for plan in plans:
        if is_applicable is not None and not is_applicable(plan):
            counts['not_applicable'] += 1
            continue
        try:
            await occ_update_one(
                client,
                vector_excludes=await vector_source_excludes(),
                doc_id=plan.crop_id,
                merger=merger_for(plan),
                index=index,
                writer_id=writer_id,
            )
        except (StaleRepairError, OCCFinalConflictError):
            counts['skipped_changed'] += 1
            continue
        except OpenSearchException:
            counts['errors'] += 1
            continue
        counts['repaired'] += 1
    return counts
