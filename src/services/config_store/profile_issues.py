"""The issue constructor of region-profile validation and the codes ``force: true``
may bypass on activation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


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


def _issue(
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
        bypassable=code in BYPASSABLE_CODES,
    )
