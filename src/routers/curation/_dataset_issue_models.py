"""Wire shape of one dataset-import issue (W10.4), shared by the preview,
the job, the issue-page route and the structured 4xx detail
(``ConfigErrorDetail.issues``). Its own module so the error-detail model
can import it without a cycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.services.curation.dataset_import.issues import (
    DatasetIssueCode,  # noqa: TC001 - pydantic field type, resolved at runtime
)


if TYPE_CHECKING:
    from src.services.curation.dataset_import.issues import DatasetIssue


class DatasetIssueSampleWire(BaseModel):
    file: str
    line: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class DatasetIssueWire(BaseModel):
    code: DatasetIssueCode
    severity: Literal['error', 'warning', 'info']
    blocking: bool
    bypassable: bool
    message: str
    count: int
    samples: list[DatasetIssueSampleWire] = Field(default_factory=list)


def issue_to_wire(issue: DatasetIssue) -> DatasetIssueWire:
    return DatasetIssueWire(
        code=issue.code,  # type: ignore[arg-type]
        severity=issue.severity,
        blocking=issue.blocking,
        bypassable=issue.bypassable,
        message=issue.message,
        count=issue.count,
        samples=[
            DatasetIssueSampleWire(file=s.file, line=s.line, detail=dict(s.detail))
            for s in issue.samples
        ],
    )


__all__ = ['DatasetIssueSampleWire', 'DatasetIssueWire', 'issue_to_wire']
