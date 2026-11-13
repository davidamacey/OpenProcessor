"""Class identity by NAME, in one place.

An item's class is identified by its ``class_name`` (registry slug) or its
``proposal_name`` (the detector's own raw label, e.g. ``traffic light``);
ids are never compared. A requested name matches either one after
:func:`~src.utils.class_names.normalize_class_name` (case, spaces, hyphens). The proposal
names an item only while it has no class: an item labelled away from the detector's guess
belongs to its class alone, so a class's list and its count are the same items.
The in-memory test (ingest policy) and the stored-item query (every filtered
list) are built from the same normalization so they cannot disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Iterable


CLASS_FIELD = 'class_name'
PROPOSAL_FIELD = 'proposal_name'
NAME_FIELDS: tuple[str, ...] = (CLASS_FIELD, PROPOSAL_FIELD)
DETECTOR_NAME_FIELDS: tuple[str, ...] = ('detector_class_name',)


def normalized_names(names: Iterable[str]) -> set[str]:
    """The distinct, non-empty normalized forms of ``names``."""
    return {n for n in (normalize_class_name(x) for x in names if x) if n}


def name_matches(
    names: Iterable[str], *, class_name: str | None, proposal_name: str | None
) -> bool:
    """Whether the item's class name, or its proposal name while it has no class, is one of
    ``names``."""
    wanted = normalized_names(names)
    named = class_name or proposal_name
    return named is not None and normalize_class_name(named) in wanted


def _spellings(name: str) -> set[str]:
    """The stored spellings one normalized name can have: the slug, the
    spaced form detectors emit, and the hyphenated form."""
    slug = normalize_class_name(name)
    return {slug, slug.replace('_', ' '), slug.replace('_', '-')} if slug else set()


def name_clause(
    names: Iterable[str], fields: tuple[str, ...] = NAME_FIELDS
) -> dict[str, Any] | None:
    """OpenSearch clause matching items whose class or proposal name (or, with
    ``fields``, the named stored fields) is one of ``names`` (case-insensitive),
    or ``None`` for an empty list (no filter)."""
    spellings = sorted({s for n in names for s in _spellings(n)})
    if not spellings:
        return None
    should: list[dict[str, Any]] = []
    for field in fields:
        for spelling in spellings:
            term: dict[str, Any] = {'term': {field: {'value': spelling, 'case_insensitive': True}}}
            if field == PROPOSAL_FIELD:
                term = {
                    'bool': {'filter': [term], 'must_not': [{'exists': {'field': CLASS_FIELD}}]}
                }
            should.append(term)
    return {'bool': {'should': should, 'minimum_should_match': 1}}
