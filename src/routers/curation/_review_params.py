"""Query-parameter declarations shared by ``GET /review/{tab}`` and its locate
twin (Annotated defaults, so direct Python callers get plain values)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Query
from pydantic import BeforeValidator


def empty_as_unset(value: object) -> object:
    """The ``Any`` option an enum/bool filter spec serves is ``""``: it means unset."""
    return None if value == '' else value


UnsetIfEmpty = BeforeValidator(empty_as_unset)


IncludeTest = Annotated[bool, Query()]
TextQ = Annotated[
    str | None,
    Query(description='Regions tab only: case-insensitive substring search on any box text.'),
]
BlurQ = Annotated[float | None, Query(ge=0.0, description='Clarity floor (null-safe).')]
MistakeQ = Annotated[float | None, Query(ge=0.0, description='Mistakenness floor (null-safe).')]
NearDupQ = Annotated[bool, Query(description='Hide non-representative near-duplicates.')]
SourceQ = Annotated[str | None, Query(description='Only items with this ingest source tag.')]
RegionStatusQ = Annotated[
    str | None,
    Query(
        description=(
            "Regions tab only (ignored elsewhere). One of 'all' (default: "
            'accepted-but-unvalidated boxes plus a verifier-rejected '
            "candidate that still has a box), 'detected', 'verify_rejected'. "
            'See GET /review/tabs filter_specs (param region_status).'
        )
    ),
]
CombineConflictQ = Annotated[
    bool,
    Query(
        description=(
            'Only items a project combine flagged: sources disagreed on the box and the '
            "first-listed source's label was kept (combine_conflict)."
        )
    ),
]
ImportIdQ = Annotated[
    str | None,
    Query(description='Imported tab only: items labeled by this dataset import.'),
]
DatasetSplitQ = Annotated[
    str | None,
    UnsetIfEmpty,
    Query(
        description=(
            'Imported tab only: the split the import filed the frame under '
            '(GET /review/tabs filter_specs, param dataset_split).'
        )
    ),
]
OnNegativeFrameQ = Annotated[
    bool | None,
    UnsetIfEmpty,
    Query(
        description=(
            'true = only items on an imported reviewed-negative frame; false = hide them '
            '(GET /review/tabs filter_specs, param on_negative_frame).'
        )
    ),
]
SortQ = Annotated[
    str | None,
    Query(
        description=(
            'Review-sort id from GET /curation/methods (axis=sort). Omitted or '
            "'default': the tab's own default. Unknown / shadow / disabled -> 400."
        )
    ),
]


__all__ = [
    'BlurQ',
    'CombineConflictQ',
    'DatasetSplitQ',
    'ImportIdQ',
    'IncludeTest',
    'MistakeQ',
    'NearDupQ',
    'OnNegativeFrameQ',
    'RegionStatusQ',
    'SortQ',
    'SourceQ',
    'TextQ',
    'UnsetIfEmpty',
]
