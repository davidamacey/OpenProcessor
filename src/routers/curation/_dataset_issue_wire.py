"""Conversion of dataset-import issues to their wire shape."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.schemas.curation.dataset_issue import DatasetIssueSampleWire, DatasetIssueWire
from src.utils.issue_ids import stamp_issue_ids


if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.services.curation.dataset_import.issues import DatasetIssue


def _issue_to_wire(issue: DatasetIssue) -> DatasetIssueWire:
    return DatasetIssueWire(
        code=issue.code,
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


__all__ = ['issues_to_wire']
