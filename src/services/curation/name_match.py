"""Class identity by NAME, in one place.

An item's class is identified by its ``class_name`` (registry slug) or its
``proposal_name`` (the detector's own raw label, e.g. ``traffic light``);
ids are never compared. A requested name matches either one after
:func:`~src.utils.class_names.normalize_class_name` (case, spaces, hyphens).
The in-memory test (ingest policy) and the stored-item query (every filtered
list) are built from the same normalization so they cannot disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Iterable


NAME_FIELDS: tuple[str, ...] = ('class_name', 'proposal_name')


def normalized_names(names: Iterable[str]) -> set[str]:
    """The distinct, non-empty normalized forms of ``names``."""
    return {n for n in (normalize_class_name(x) for x in names if x) if n}


def name_matches(
    names: Iterable[str], *, class_name: str | None, proposal_name: str | None
) -> bool:
    """Whether the item's class or proposal name is one of ``names``."""
    wanted = normalized_names(names)
    return any(
        normalize_class_name(candidate) in wanted
        for candidate in (class_name, proposal_name)
        if candidate
    )


def _spellings(name: str) -> set[str]:
    """The stored spellings one normalized name can have: the slug, the
    spaced form detectors emit, and the hyphenated form."""
    slug = normalize_class_name(name)
    return {slug, slug.replace('_', ' '), slug.replace('_', '-')} if slug else set()


def name_clause(names: Iterable[str]) -> dict[str, Any] | None:
    """OpenSearch clause matching items whose class or proposal name is one of
    ``names`` (case-insensitive), or ``None`` for an empty list (no filter)."""
    spellings = sorted({s for n in names for s in _spellings(n)})
    if not spellings:
        return None
    return {
        'bool': {
            'should': [
                {'term': {field: {'value': spelling, 'case_insensitive': True}}}
                for field in NAME_FIELDS
                for spelling in spellings
            ],
            'minimum_should_match': 1,
        }
    }
