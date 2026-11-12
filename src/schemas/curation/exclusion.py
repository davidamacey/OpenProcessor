"""Why an item is excluded: the vocabulary of `POST /crops/batch_exclude` and `GET /exclude_reasons`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


ExcludeReason = Literal[
    'ignore', 'blurry', 'unidentifiable', 'not_the_subject', 'partial_crop', 'not_an_object'
]

EXCLUDE_REASON_LABELS: dict[ExcludeReason, str] = {
    'ignore': 'Ignore (generic)',
    'blurry': 'Blurry',
    'unidentifiable': 'Unidentifiable',
    'not_the_subject': 'Not the subject',
    'partial_crop': 'Partial crop',
    'not_an_object': 'Not an object (delete the box)',
}


class ExcludeReasonEntry(BaseModel):
    id: ExcludeReason
    label: str


class ExcludeReasonList(BaseModel):
    reasons: list[ExcludeReasonEntry]


def exclude_reason_list() -> ExcludeReasonList:
    return ExcludeReasonList(
        reasons=[
            ExcludeReasonEntry(id=r, label=label) for r, label in EXCLUDE_REASON_LABELS.items()
        ]
    )
