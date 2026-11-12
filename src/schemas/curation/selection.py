"""Wire models of the selection summary: what a bulk action on a selection would touch."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassCount(BaseModel):
    class_name: str = Field(description='A class name, `unlabeled` or `other` (the rest).')
    count: int


class SelectionSummary(BaseModel):
    detections: int
    images: int
    clusters: int
    locked: int = Field(
        description='Detections a bulk write skips: a person or an import set their class or '
        'boxes, or they sit in the test holdout.'
    )
    in_holdout: int
    by_class: list[ClassCount] = Field(description='The 10 largest classes, then `other`.')
    taken_from: str | None = None
    taken_to: str | None = None
    located: int = Field(description='Images with a position the location policy lets be served.')
    join_truncated: bool = Field(description='More images matched than the position count used.')
    summary_text: str = Field(description='"183 detections in 41 images · 7 locked, skipped".')
