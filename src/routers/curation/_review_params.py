"""Query-parameter declarations shared by ``GET /review/{tab}`` and its locate
twin (Annotated defaults, so direct Python callers get plain values)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Query


IncludeTest = Annotated[bool, Query()]
TextQ = Annotated[
    str | None,
    Query(description='Regions tab only: case-insensitive substring search on any box text.'),
]
MaxRankQ = Annotated[
    int | None,
    Query(
        ge=1,
        description=(
            'Keep crop_rank_in_image <= this (every tab). Omitted: no limit, '
            "except a tab's served filter_defaults (GET /review/tabs)."
        ),
    ),
]
BlurQ = Annotated[float | None, Query(ge=0.0, description='Clarity floor (null-safe).')]
MistakeQ = Annotated[float | None, Query(ge=0.0, description='Mistakenness floor (null-safe).')]
NearDupQ = Annotated[bool, Query(description='Hide non-representative near-duplicates.')]
ClassIdQ = Annotated[int | None, Query(description='Only items of this class.')]
SourceQ = Annotated[str | None, Query(description='Only items with this ingest source tag.')]
ConfQ = Annotated[float | None, Query(ge=0.0, le=1.0, description='Inclusive confidence band.')]
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
    Query(
        description=(
            'Imported tab only: the split the import filed the frame under '
            '(GET /review/tabs filter_specs, param dataset_split).'
        )
    ),
]
OnNegativeFrameQ = Annotated[
    bool | None,
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
    'ClassIdQ',
    'CombineConflictQ',
    'ConfQ',
    'DatasetSplitQ',
    'ImportIdQ',
    'IncludeTest',
    'MaxRankQ',
    'MistakeQ',
    'NearDupQ',
    'OnNegativeFrameQ',
    'RegionStatusQ',
    'SortQ',
    'SourceQ',
    'TextQ',
]
