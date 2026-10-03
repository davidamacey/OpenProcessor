"""The served reprocess vocabulary covers exactly what the backend accepts
and writes, so a new scope / filter field / status can't ship unlabelled."""

from __future__ import annotations

from typing import get_args

from src.services.curation.reprocess_models import ReprocessFilter, ReprocessScope
from src.services.curation.reprocess_vocabulary import (
    FILTER_FIELD_LABELS,
    JOB_STATUS_LABELS,
    LOCK_REASON_LABELS,
    SCOPE_LABELS,
    lock_reason_vocabulary,
    scope_vocabulary,
)


def test_every_scope_is_labelled_and_nothing_extra() -> None:
    assert set(SCOPE_LABELS) == set(get_args(ReprocessScope))
    assert [e.id for e in scope_vocabulary()] == list(get_args(ReprocessScope))


def test_filter_labels_cover_the_reprocess_only_selectors() -> None:
    from src.services.curation.item_filter import ItemFilter

    own = set(ReprocessFilter.model_fields) - set(ItemFilter.model_fields)
    assert own == set(FILTER_FIELD_LABELS)


def test_job_statuses_are_the_ones_the_runner_writes() -> None:
    import inspect

    from src.services.curation import reprocess_job

    src = inspect.getsource(reprocess_job)
    for status in JOB_STATUS_LABELS:
        assert f"'{status}'" in src, status


def test_lock_reasons_have_human_text() -> None:
    assert {'human_label', 'validated', 'imported'} <= set(LOCK_REASON_LABELS)
    assert all(e.description for e in lock_reason_vocabulary())
