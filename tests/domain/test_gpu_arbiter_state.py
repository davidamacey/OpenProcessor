"""Who pauses the VLM workers: a card-holding claim, or a live ``vlm use`` model swap."""

from __future__ import annotations

from src.domain import gpu_arbiter_state as state


NOW = 1_000_000.0


def test_an_empty_or_missing_document_pauses_nothing() -> None:
    assert not state.is_paused(None, now=NOW)
    assert not state.is_paused({}, now=NOW)
    assert not state.is_paused({state.ARBITER: state.ArbiterRecord().to_json()}, now=NOW)


def test_a_claim_pauses_the_workers_until_it_is_gone() -> None:
    held = {state.ARBITER: state.ArbiterRecord(claims=('job-1',)).to_json()}
    assert state.is_paused(held, now=NOW)
    # a record that only still lists a stopped service is not a claim: the run has ended
    assert not state.is_paused(
        {state.ARBITER: state.ArbiterRecord(stopped=('vlm',)).to_json()}, now=NOW
    )


def test_a_fresh_model_swap_pauses_the_workers() -> None:
    assert state.is_paused({state.VLM_SWITCH: {'at': NOW - 60}}, now=NOW)


def test_a_swap_that_crashed_stops_counting_after_the_ttl() -> None:
    stale = {state.VLM_SWITCH: {'at': NOW - state.SWITCH_TTL_S - 1}}
    assert not state.is_paused(stale, now=NOW)
    assert state.is_paused({state.VLM_SWITCH: {'at': NOW - state.SWITCH_TTL_S + 1}}, now=NOW)


def test_a_malformed_document_is_not_a_pause() -> None:
    """The key's CHECK allows any jsonb; a stray scalar must not crash a worker."""
    assert not state.is_paused(True, now=NOW)
    assert not state.is_paused({state.ARBITER: 'x', state.VLM_SWITCH: 3}, now=NOW)


def test_the_arbiter_record_round_trips() -> None:
    record = state.ArbiterRecord(
        claims=('a', 'b'), stopped=('vlm',), acknowledged=('a',), error='stop vlm: refused', at=5.0
    )
    assert state.arbiter_record({state.ARBITER: record.to_json()}) == record
