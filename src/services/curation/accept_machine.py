"""Accept machine labels: a person validates the machine's class on many items at once.

The detector's and the VLM's labels sit in "Check machine labels" until someone confirms them one
by one. This counts what a bar (``min_confidence``, ``sources``, an item filter) would accept, by
class and with what is held back and why, and queues the recorded label operation that does it
(kind ``accept_machine``, undone like any label operation). The per-item merge and the rule of
which labels qualify are in :mod:`~src.services.curation.label_ops.accept_merge`.

Never written: an item a person or an import validated, one in the test holdout, an excluded or
review-dismissed one, one drawn into an open accuracy audit, and one whose class is deprecated or
the region class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.domain.audit_math import AUDIT_SAMPLE
from src.domain.ingest_class_sources import HUMAN_BULK_ACCEPT_LABEL_SOURCE
from src.services.curation.image_join import resolve_filter_join
from src.services.curation.item_filter import item_filter_clauses
from src.services.curation.item_selection import MAX_BULK_SELECTION, resolve_ids
from src.services.curation.label_ops import record as label_op_record
from src.services.curation.label_ops.accept_merge import ACCEPT_WRITER, accept_clause
from src.services.curation.label_ops.store import NewLabelOp
from src.services.curation.region_class import item_classes
from src.services.embedding_spaces.readers import item_write_fields
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    import uuid
    from collections.abc import Collection, Iterable

    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.clients.curation_opensearch import RegistryClassEntry
    from src.services.curation.item_filter import ItemFilter

logger = get_logger(__name__)

MAX_CLASSES = 5000
"""Distinct classes one count reads."""

_EXCLUDED: list[dict[str, Any]] = [
    {'term': {'class_excluded': True}},
    {'exists': {'field': 'review_dismissed_at'}},
]
_HOLDOUT = {'term': {'test_holdout': True}}
_AUDIT = {'term': {AUDIT_SAMPLE: True}}
_VALIDATED = {'term': {'class_validated': True}}


@dataclass(frozen=True)
class HeldBack:
    """Items the bar matches that the operation will not write, by the first reason that applies
    (excluded or dismissed, then holdout, open audit sample, validated, deprecated class)."""

    excluded: int
    holdout: int
    audit_sample: int
    locked: int
    deprecated_class: int


@dataclass(frozen=True)
class AcceptPlan:
    """What accepting would do."""

    by_class: tuple[tuple[str, int], ...]
    """``(class_name, items)`` per active class, most items first."""
    held_back: HeldBack
    classes: dict[str, tuple[int, str]]
    """The active registry classes the items carry, by name (registry-name form), to
    ``(class_id, class_name)``: the only classes the merge may write."""
    query: dict[str, Any]
    """The items the operation writes, before the cap."""

    @property
    def items(self) -> int:
        return sum(n for _, n in self.by_class)


def active_classes(classes: Iterable[RegistryClassEntry]) -> dict[str, tuple[int, str]]:
    """The classes a whole item may be given: active, and not the region class."""
    return {
        normalize_class_name(c.class_name): (c.class_id, c.class_name)
        for c in item_classes(classes)
    }


async def _base_clauses(
    opensearch: AsyncOpenSearch,
    *,
    min_confidence: float,
    sources: Collection[str],
    item_filter: ItemFilter | None,
) -> list[dict[str, Any]]:
    """The bar and the caller's filter, with no protection applied (those are counted apart)."""
    clauses = [accept_clause(min_confidence, sources)]
    if item_filter is not None:
        join = await resolve_filter_join(opensearch, item_filter)
        clauses += item_filter_clauses(
            item_filter, embedding=await item_write_fields(), image_join=join
        )
    return clauses


def _bool(filter_: list[dict[str, Any]], must_not: list[dict[str, Any]]) -> dict[str, Any]:
    return {'bool': {'filter': filter_, 'must_not': must_not}}


async def plan_accept(
    opensearch: AsyncOpenSearch,
    classes: Iterable[RegistryClassEntry],
    *,
    min_confidence: float,
    sources: Collection[str],
    item_filter: ItemFilter | None,
) -> AcceptPlan:
    """Count what the bar would accept and what it holds back, by one search and four counts."""
    active = active_classes(classes)
    base = await _base_clauses(
        opensearch, min_confidence=min_confidence, sources=sources, item_filter=item_filter
    )
    # The protections in the order they are reported, so no item is counted twice.
    excluded = {'bool': {'should': _EXCLUDED, 'minimum_should_match': 1}}
    writable = _bool(base, [excluded, _HOLDOUT, _AUDIT, _VALIDATED])
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 0,
            'query': writable,
            'aggs': {'by_class': {'terms': {'field': 'class_name', 'size': MAX_CLASSES}}},
        },
    )
    buckets = ((resp.get('aggregations') or {}).get('by_class') or {}).get('buckets') or []
    counts = {str(b['key']): int(b['doc_count']) for b in buckets}
    if (resp.get('aggregations') or {}).get('by_class', {}).get('sum_other_doc_count'):
        logger.warning('accept_machine_truncated', classes=MAX_CLASSES)
    named = {name: active[name] for name in counts if name in active}
    by_class = sorted(
        ((class_name, counts[name]) for name, (_id, class_name) in named.items()),
        key=lambda kv: (-kv[1], kv[0]),
    )
    deprecated = sum(n for name, n in counts.items() if name not in active)

    async def held(clauses: list[dict[str, Any]], not_yet: list[dict[str, Any]]) -> int:
        query = _bool([*base, *clauses], not_yet)
        return int((await opensearch.count(index=items_index(), body={'query': query}))['count'])

    held_back = HeldBack(
        excluded=await held([excluded], []),
        holdout=await held([_HOLDOUT], [excluded]),
        audit_sample=await held([_AUDIT], [excluded, _HOLDOUT]),
        locked=await held([_VALIDATED], [excluded, _HOLDOUT, _AUDIT]),
        deprecated_class=deprecated,
    )
    query = {'bool': {'filter': [writable, {'terms': {'class_name': sorted(named)}}]}}
    return AcceptPlan(tuple(by_class), held_back, named, query)


@dataclass(frozen=True)
class AcceptStart:
    """What :func:`start_accept` did."""

    plan: AcceptPlan
    op_id: uuid.UUID | None
    job_id: uuid.UUID | None
    """The recorded operation and its job; ``None`` when nothing qualified."""
    remaining: int
    """Items beyond this operation (it writes at most ``MAX_BULK_SELECTION``): accept again."""


async def start_accept(
    session: AsyncSession,
    project: uuid.UUID,
    opensearch: AsyncOpenSearch,
    classes: Iterable[RegistryClassEntry],
    *,
    min_confidence: float,
    sources: Collection[str],
    item_filter: ItemFilter | None,
) -> AcceptStart:
    """Queue the recorded operation that accepts the labels :func:`plan_accept` counted."""
    plan = await plan_accept(
        opensearch, classes, min_confidence=min_confidence, sources=sources, item_filter=item_filter
    )
    ids: list[str] = []
    if plan.items:
        ids = await resolve_ids(
            opensearch, plan.query, index=items_index(), limit=MAX_BULK_SELECTION
        )
    if not ids:
        return AcceptStart(plan, None, None, 0)
    op_id, job = await label_op_record.start(
        session,
        project,
        op=NewLabelOp(
            kind='accept_machine',
            writer=ACCEPT_WRITER,
            class_name=None,
            class_id=None,
            label_source=HUMAN_BULK_ACCEPT_LABEL_SOURCE,
            target='selection',
            selected=len(ids),
        ),
        spec={
            'kind': 'accept_machine',
            'writer': ACCEPT_WRITER,
            'min_confidence': min_confidence,
            'sources': sorted(sources),
            'classes': {name: list(v) for name, v in plan.classes.items()},
        },
        ids=ids,
    )
    return AcceptStart(plan, op_id, job.id, max(0, plan.items - len(ids)))


__all__ = [
    'MAX_CLASSES',
    'AcceptPlan',
    'AcceptStart',
    'HeldBack',
    'active_classes',
    'plan_accept',
    'start_accept',
]
