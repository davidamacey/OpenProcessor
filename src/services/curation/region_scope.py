"""Which items get the region stage: a region profile's ``parent_classes``.

An empty set means every item. Otherwise an item is in scope when its
``class_name`` or its ``proposal_name`` (the ingest proposer's raw class
name, kept even after a relabel) names one of the parent classes,
compared case-insensitively. Ingest seeds only in-scope items
(:func:`~src.services.curation.item_doc.region_seed_status`) and the
detection worker only fetches them (:func:`parent_classes_clause`), so a
profile change also stops items seeded before it from being processed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterable


PARENT_CLASS_FIELDS: tuple[str, ...] = ('class_name', 'proposal_name')


def _normalized(names: Iterable[str]) -> list[str]:
    return sorted({n.strip().lower() for n in names if n and n.strip()})


def in_parent_classes(
    parent_classes: Iterable[str], *, class_name: str | None, proposal_name: str | None
) -> bool:
    """Whether an item with these names gets the region stage."""
    wanted = set(_normalized(parent_classes))
    if not wanted:
        return True
    return any(
        name is not None and name.strip().lower() in wanted for name in (class_name, proposal_name)
    )


def parent_classes_clause(parent_classes: Iterable[str]) -> dict[str, Any] | None:
    """OpenSearch filter clause matching in-scope items, or ``None`` (no
    filter) when every item is in scope.

    ``term`` with ``case_insensitive`` (OpenSearch 1.0+) instead of a
    ``terms`` query, which has no case-insensitive form: stored class and
    proposal names keep whatever case their writer used.
    """
    names = _normalized(parent_classes)
    if not names:
        return None
    return {
        'bool': {
            'should': [
                {'term': {field: {'value': name, 'case_insensitive': True}}}
                for field in PARENT_CLASS_FIELDS
                for name in names
            ],
            'minimum_should_match': 1,
        }
    }


__all__ = ['PARENT_CLASS_FIELDS', 'in_parent_classes', 'parent_classes_clause']
