"""Adopt detector classes: give the items the detector named but nobody classed that class.

An item the detector found keeps the detector's own label (``detector_class_name``). Items stored
before ingest assigned it, or stored while the registry lacked the class, have no ``class_id``.
Adoption writes the class by NAME (class identity is the name, never the detector's index) as the
detector's machine label (:data:`~src.domain.ingest_class_sources.DETECTOR_CLASS_SOURCE`, never
validated) and creates the registry classes it needs. It runs as a recorded label operation, so it
shows in Label history and can be undone.

Never written: an item with a class, a validated or human/import-locked one, a test-holdout item, a
human-excluded one. A detector name that matches a deprecated class or the region class is skipped
and reported (the rule ``seed_from_detector`` follows), never brought back or duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.services.curation.detector_class_map import ensure_detector_class
from src.services.curation.item_selection import MAX_BULK_SELECTION, resolve_ids
from src.services.curation.label_ops import record as label_op_record
from src.services.curation.label_ops.adopt_merge import ADOPT_WRITER
from src.services.curation.label_ops.store import NewLabelOp
from src.services.curation.region_class import item_classes
from src.utils.class_names import class_name_deprecation_index, normalize_class_name


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable

    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.clients.curation_opensearch import ClassRegistryFile, RegistryClassEntry

logger = get_logger(__name__)

MAX_DETECTOR_NAMES = 5000
"""Distinct detector labels one census reads (an LVIS-sized vocabulary is about 1,200)."""


def adoptable_query() -> dict[str, Any]:
    """Items the detector named that have no class and that no one has locked."""
    return {
        'bool': {
            'filter': [{'exists': {'field': 'detector_class_name'}}],
            'must_not': [
                {'exists': {'field': 'class_id'}},
                {'term': {'class_validated': True}},
                {'term': {'class_excluded': True}},
                {'term': {'test_holdout': True}},
            ],
        }
    }


@dataclass(frozen=True)
class DetectorCensus:
    """The detector labels the items carry, by name (registry-name form)."""

    seen: dict[str, int]
    """Items per detector name, every item the detector named."""
    adoptable: dict[str, int]
    """Of those, the items :func:`adoptable_query` matches."""


@dataclass(frozen=True)
class AdoptionPlan:
    """What adoption would do, by detector name (registry-name form)."""

    create: tuple[str, ...]
    """Names the registry lacks: a class is added for each, so that every label the detector
    produced is in the registry."""
    existing: dict[str, tuple[int, str]]
    """Names an active registry class already carries: ``(class_id, class_name)``."""
    blocked: tuple[str, ...]
    """Names of a deprecated class or the region class: skipped."""
    adoptable: dict[str, int]
    """Adoptable items per detector name."""

    @property
    def items(self) -> int:
        """Items adoption would give a class."""
        return sum(n for name, n in self.adoptable.items() if name not in self.blocked)

    @property
    def blocked_items(self) -> int:
        """Adoptable items whose detector name is blocked: left without a class."""
        return sum(n for name, n in self.adoptable.items() if name in self.blocked)


def plan_adoption(census: DetectorCensus, classes: Iterable[RegistryClassEntry]) -> AdoptionPlan:
    """The plan for ``census`` against the registry's ``classes``."""
    classes = list(classes)
    known = class_name_deprecation_index(classes)
    active = {
        normalize_class_name(c.class_name): (c.class_id, c.class_name)
        for c in item_classes(classes)
    }
    create: list[str] = []
    existing: dict[str, tuple[int, str]] = {}
    blocked: list[str] = []
    for name in sorted(census.seen):
        slug = normalize_class_name(name)
        if slug in active:
            existing[name] = active[slug]
        elif slug in known:
            blocked.append(name)
        elif slug:
            create.append(name)
    return AdoptionPlan(tuple(create), existing, tuple(blocked), dict(census.adoptable))


async def detector_census(opensearch: AsyncOpenSearch) -> DetectorCensus:
    """Both counts per detector name, from one search."""
    names = {'terms': {'field': 'detector_class_name', 'size': MAX_DETECTOR_NAMES}}
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 0,
            'query': {'exists': {'field': 'detector_class_name'}},
            'aggs': {
                'seen': names,
                'adoptable': {'filter': adoptable_query(), 'aggs': {'names': names}},
            },
        },
    )
    aggs = resp.get('aggregations') or {}
    seen = aggs.get('seen') or {}
    adoptable = (aggs.get('adoptable') or {}).get('names') or {}
    if seen.get('sum_other_doc_count') or adoptable.get('sum_other_doc_count'):
        logger.warning('detector_census_truncated', names=MAX_DETECTOR_NAMES)

    def counts(agg: dict[str, Any]) -> dict[str, int]:
        return {str(b['key']): int(b['doc_count']) for b in agg.get('buckets') or []}

    return DetectorCensus(counts(seen), counts(adoptable))


@dataclass(frozen=True)
class AdoptionStart:
    """What :func:`start_adoption` did."""

    plan: AdoptionPlan
    created: tuple[str, ...]
    """Classes added to the registry."""
    op_id: uuid.UUID | None
    job_id: uuid.UUID | None
    """The recorded operation and its job; ``None`` when no item was left to label."""
    remaining: int
    """Adoptable items this operation does not cover (an operation writes at most
    ``MAX_BULK_SELECTION`` items): adopt again for the rest."""


async def start_adoption(
    session: AsyncSession,
    project: uuid.UUID,
    opensearch: AsyncOpenSearch,
    registry: ClassRegistryFile,
) -> AdoptionStart:
    """Add the registry classes the detector's labels need, then queue the recorded operation
    that gives the unclassed items their detector's class."""
    plan = plan_adoption(await detector_census(opensearch), registry.classes)
    classes = dict(plan.existing)
    for name in plan.create:
        classes[name] = await ensure_detector_class(normalize_class_name(name), name)
    ids: list[str] = []
    if classes and plan.items:
        query = adoptable_query()
        query['bool']['filter'].append({'terms': {'detector_class_name': sorted(classes)}})
        ids = await resolve_ids(opensearch, query, index=items_index(), limit=MAX_BULK_SELECTION)
    if not ids:
        return AdoptionStart(plan, plan.create, None, None, 0)
    op_id, job = await label_op_record.start(
        session,
        project,
        op=NewLabelOp(
            kind='adopt_detector',
            writer=ADOPT_WRITER,
            class_name=None,
            class_id=None,
            target='selection',
            selected=len(ids),
        ),
        spec={
            'kind': 'adopt_detector',
            'writer': ADOPT_WRITER,
            'classes': {normalize_class_name(n): list(v) for n, v in classes.items()},
        },
        ids=ids,
    )
    return AdoptionStart(plan, plan.create, op_id, job.id, max(0, plan.items - len(ids)))


__all__ = [
    'MAX_DETECTOR_NAMES',
    'AdoptionPlan',
    'AdoptionStart',
    'DetectorCensus',
    'adoptable_query',
    'detector_census',
    'plan_adoption',
    'start_adoption',
]
