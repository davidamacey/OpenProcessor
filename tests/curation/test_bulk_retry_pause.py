"""What a failed file's ledger row says: terminal kinds are dead at once, transient kinds keep their
attempts and wait (the kind and the API's ``Retry-After`` decide how long, with a cap), any other
kind counts one attempt. The SQL that applies it is tested on PostgreSQL in
``test_ingest_source_files_claim.py``."""

from __future__ import annotations

import pytest

from src.services.curation.bulk_ingest import source_queue as queue
from src.services.curation.ingest_models import (
    ERROR_KIND_BUSY,
    ERROR_KIND_DATABASE_BUSY,
    ERROR_KIND_ENGINE_UNAVAILABLE,
    ERROR_KIND_INCOMPLETE_WRITE,
    TRANSIENT_ERROR_KINDS,
)


def test_every_transient_kind_has_a_pause_and_no_other_kind_has_one() -> None:
    assert set(queue.RETRY_PAUSE_S) == set(TRANSIENT_ERROR_KINDS)
    assert {
        ERROR_KIND_ENGINE_UNAVAILABLE,
        ERROR_KIND_DATABASE_BUSY,
        ERROR_KIND_INCOMPLETE_WRITE,
        ERROR_KIND_BUSY,
    } == TRANSIENT_ERROR_KINDS
    assert not TRANSIENT_ERROR_KINDS & queue.TERMINAL_ERROR_KINDS
    assert all(0 < s <= queue.MAX_RETRY_PAUSE_S for s in queue.RETRY_PAUSE_S.values())


@pytest.mark.parametrize('kind', sorted(TRANSIENT_ERROR_KINDS))
def test_a_transient_failure_is_not_terminal_and_waits(kind: str) -> None:
    row = queue.failure_row('f', 'why', kind)
    assert row.terminal is False
    assert row.retry_in_s == queue.RETRY_PAUSE_S[kind]
    assert row.error == f'[{kind}] why'


@pytest.mark.parametrize('kind', ['batch_error', 'bulk_index', 'fetch', '', 'something_new'])
def test_any_other_kind_counts_an_attempt_and_is_claimable_at_once(kind: str) -> None:
    row = queue.failure_row('f', 'why', kind, retry_after_s=90.0)
    assert (row.terminal, row.retry_in_s) == (False, None)  # Retry-After changes nothing here


@pytest.mark.parametrize('kind', sorted(queue.TERMINAL_ERROR_KINDS))
def test_a_terminal_kind_is_dead_at_once_and_never_waits(kind: str) -> None:
    row = queue.failure_row('f', 'why', kind, retry_after_s=90.0)
    assert (row.terminal, row.retry_in_s) == (True, None)


def test_retry_after_raises_a_pause_and_never_shortens_it() -> None:
    assert queue.retry_pause_s(ERROR_KIND_BUSY, 120.0) == 120.0
    assert queue.retry_pause_s(ERROR_KIND_BUSY, 5.0) == queue.RETRY_PAUSE_S[ERROR_KIND_BUSY]
    assert queue.retry_pause_s(ERROR_KIND_BUSY, 0.0) == queue.RETRY_PAUSE_S[ERROR_KIND_BUSY]


def test_no_pause_passes_the_cap_however_long_the_api_asked() -> None:
    assert queue.retry_pause_s(ERROR_KIND_BUSY, 86_400.0) == queue.MAX_RETRY_PAUSE_S
    assert queue.retry_pause_s(ERROR_KIND_INCOMPLETE_WRITE, 86_400.0) == queue.MAX_RETRY_PAUSE_S
