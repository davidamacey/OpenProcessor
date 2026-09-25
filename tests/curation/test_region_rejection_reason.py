"""R10: a rejected region's per-item reason must not say both "rejected"
and "needs human review" about the same box.

Live evidence: a ``verify_rejected`` item with ``region_rejection_reason
= 'verifier_no_verdict'`` served the status label "rejected (bad
detection)" alongside the per-item reason "verifier rejected this
candidate (verifier_no_verdict) — needs human review" -- a raw id
embedded verbatim, plus a self-contradicting "rejected ... needs human
review" sentence. :func:`compose_rejection_reason` words the reason from
the served rejection-reason vocabulary (kind ``model_verdict`` /
``automatic`` / ``needs_human``) and only one verb ever appears.
"""

from __future__ import annotations

from src.config.region_rejection import (
    REJECT_REASON_NO_VERDICT,
    REJECT_REASON_SANITY_PREFIX,
    REJECT_REASON_VERIFIER,
    compose_rejection_reason,
)
from src.services.curation.review_queries import region_reason


def test_no_verdict_reads_needs_human_review_not_rejected() -> None:
    text = compose_rejection_reason(REJECT_REASON_NO_VERDICT)
    assert text == 'needs human review: verifier gave no verdict'
    assert 'rejected' not in text


def test_model_verdict_reason_reads_rejected() -> None:
    text = compose_rejection_reason(REJECT_REASON_VERIFIER)
    assert text.startswith('rejected:')
    assert 'needs human review' not in text


def test_sanity_prefix_reason_keeps_the_detail_and_reads_rejected() -> None:
    text = compose_rejection_reason(f'{REJECT_REASON_SANITY_PREFIX}degenerate_zero_size')
    assert text.startswith('rejected:')
    assert 'degenerate_zero_size' in text
    assert 'needs human review' not in text


def test_unrecognized_reason_falls_back_to_the_raw_value() -> None:
    text = compose_rejection_reason('a human free-text reason')
    assert text == 'rejected: a human free-text reason'


def test_region_reason_for_no_verdict_item_never_says_rejected() -> None:
    """The exact live scenario: a verify_rejected item with a recorded
    no-verdict rejection reason must not have its per-item reason claim
    "rejected" anywhere."""

    class _Fields:
        status = 'region_status'
        rejection_reason = 'region_rejection_reason'

    src = {
        'region_status': 'verify_rejected',
        'region_rejection_reason': REJECT_REASON_NO_VERDICT,
    }
    text = region_reason(src, _Fields(), default='unused')
    assert text == 'needs human review: verifier gave no verdict'
    assert 'rejected' not in text


def test_region_reason_for_model_verdict_item_says_rejected_only() -> None:
    class _Fields:
        status = 'region_status'
        rejection_reason = 'region_rejection_reason'

    src = {
        'region_status': 'verify_rejected',
        'region_rejection_reason': REJECT_REASON_VERIFIER,
    }
    text = region_reason(src, _Fields(), default='unused')
    assert text.startswith('rejected:')
    assert 'needs human review' not in text


def test_region_reason_passes_through_default_when_not_rejected() -> None:
    class _Fields:
        status = 'region_status'
        rejection_reason = 'region_rejection_reason'

    src = {'region_status': 'detected'}
    assert region_reason(src, _Fields(), default='some other reason') == 'some other reason'
