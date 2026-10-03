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
from src.services.issue_ids import stamp_issue_ids


if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.services.curation.dataset_import.issues import DatasetIssue


class DatasetIssueSampleWire(BaseModel):
    file: str
    line: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class DatasetIssueWire(BaseModel):
    id: str = Field(
        default='',
        description='Unique within the response (`code[:subject]`, `#2` on a repeat).',
    )
    code: DatasetIssueCode
    severity: Literal['error', 'warning', 'info']
    blocking: bool
    bypassable: bool
    message: str
    count: int
    samples: list[DatasetIssueSampleWire] = Field(default_factory=list)


def _issue_to_wire(issue: DatasetIssue) -> DatasetIssueWire:
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


def _subject(issue: DatasetIssueWire) -> str | None:
    first = issue.samples[0] if issue.samples else None
    if first is None:
        return None
    return str(first.detail.get('class') or first.file or '') or None


def issues_to_wire(issues: Iterable[DatasetIssue]) -> list[DatasetIssueWire]:
    """The wire list for one response, every issue carrying a unique ``id``."""
    wired = [_issue_to_wire(i) for i in issues]
    stamp_issue_ids(wired, _subject)
    return wired


__all__ = ['DatasetIssueSampleWire', 'DatasetIssueWire', 'issues_to_wire']
