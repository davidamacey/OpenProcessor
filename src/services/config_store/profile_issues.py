"""The issue constructor of region-profile validation and the codes ``force: true``
may bypass on activation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from src.schemas.curation.config_common import ValidationIssue

#: Errors this validator raises that ``force: true`` bypasses on
#: activation -- unreachable/not-ready dependency checks that are, by
#: their nature, sometimes stale or racy; everything else (field
#: decode/range errors, no_candidate_source, the multi-box pairing
#: error) is never bypassable.
BYPASSABLE_CODES: frozenset[str] = frozenset(
    {
        'detector_model_not_ready',
        'triton_unreachable',
        'segmenter_unreachable',
        'ocr_model_not_ready',
    }
)


class IssueBuilder(Protocol):
    """Builds one :class:`ValidationIssue`, stamping ``bypassable`` from its validator's own set."""

    def __call__(
        self,
        code: str,
        severity: str,
        message: str,
        *,
        field: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> ValidationIssue: ...


def issue_builder(bypass_codes: frozenset[str]) -> IssueBuilder:
    """The issue constructor of one validator. ``bypass_codes`` is that validator's own
    ``BYPASSABLE_CODES``, required (no default), so the per-issue flag always matches its
    ``force_allowed``. A check helper shared by several validators takes the builder as a
    parameter instead of building issues itself."""

    def build(
        code: str,
        severity: str,
        message: str,
        *,
        field: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> ValidationIssue:
        from src.schemas.curation.config_common import ValidationIssue

        return ValidationIssue(
            code=code,
            severity=severity,
            field=field,
            message=message,
            detail=detail or {},
            bypassable=code in bypass_codes,
        )

    return build


#: The region-profile validator's builder (its own ``BYPASSABLE_CODES``).
region_issue: IssueBuilder = issue_builder(BYPASSABLE_CODES)
