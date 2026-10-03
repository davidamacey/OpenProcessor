"""Every issue list a response serves carries ids unique within that response."""

from __future__ import annotations

from src.routers.curation._config_common_models import ValidationIssue, ValidationReport
from src.routers.curation._dataset_issue_models import issues_to_wire
from src.services.curation.dataset_import.issues import DatasetIssue, DatasetIssueSample
from src.services.issue_ids import stamp_issue_ids
from src.services.projects.combine.models import CombineIssue, CombinePreview


def _unique(ids: list[str]) -> None:
    assert all(ids)
    assert len(set(ids)) == len(ids)


def test_stamp_suffixes_a_repeat_and_is_stable() -> None:
    class _I:
        def __init__(self, code: str) -> None:
            self.code, self.id = code, ''

    items = [_I('a'), _I('a'), _I('b')]
    stamp_issue_ids(items, lambda _i: None)
    assert [i.id for i in items] == ['a', 'a#2', 'b']


def test_validation_report_ids_are_unique_across_errors_and_warnings() -> None:
    def issue(code: str, field: str | None, severity: str = 'error') -> ValidationIssue:
        return ValidationIssue(code=code, severity=severity, field=field, message='m')  # type: ignore[arg-type]

    report = ValidationReport(
        ok=False,
        errors=[
            issue('pack_name_invalid', 'targets[0]'),
            issue('pack_name_invalid', 'targets[1]'),
            issue('pack_name_invalid', 'targets[1]'),
        ],
        warnings=[issue('pack_name_invalid', 'targets[0]', 'warning')],
    )
    ids = [i.id for i in [*report.errors, *report.warnings]]
    _unique(ids)
    assert ids[0] == 'pack_name_invalid:targets[0]'


def test_combine_preview_ids_distinguish_two_unmapped_classes() -> None:
    preview = CombinePreview(
        ok=False,
        errors=[
            CombineIssue(code='unmapped_class', project='a', detail={'class': 'bus'}),
            CombineIssue(code='unmapped_class', project='a', detail={'class': 'van'}),
            CombineIssue(code='unmapped_class', project='b', detail={'class': 'bus'}),
        ],
        warnings=[],
        preview_sha='x',
        suggested_mapping={},
        sources=[],
        target={},
        dedup={},
        bytes={},
    )
    ids = [i.id for i in preview.errors]
    _unique(ids)
    assert ids == [
        'unmapped_class:a:bus',
        'unmapped_class:a:van',
        'unmapped_class:b:bus',
    ]


def test_dataset_issue_ids_are_unique_for_a_repeated_code() -> None:
    def issue(file: str) -> DatasetIssue:
        return DatasetIssue(
            code='class_unmapped',
            severity='error',
            blocking=True,
            bypassable=False,
            message='m',
            count=1,
            samples=[DatasetIssueSample(file=file, detail={})],
        )

    wired = issues_to_wire([issue('a.txt'), issue('a.txt'), issue('b.txt')])
    _unique([w.id for w in wired])
