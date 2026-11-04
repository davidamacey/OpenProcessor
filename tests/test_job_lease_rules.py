"""Pure rules of the job lease: expiry, document ids, generations."""

from __future__ import annotations

from src.domain import job_lease as jl


def _lease(**over: object) -> dict[str, float | str]:
    base: dict[str, float | str] = {
        'heartbeat_at': 1000.0,
        'ttl_s': 120.0,
        'state': 'held',
        'generation': 3,
    }
    base.update(over)  # type: ignore[arg-type]
    return base


def test_expiry_is_exactly_heartbeat_plus_ttl() -> None:
    assert not jl.is_expired(_lease(), 1120.0)
    assert jl.is_expired(_lease(), 1120.01)


def test_released_lease_is_takeable_while_fresh() -> None:
    assert jl.is_takeable(_lease(state='released'), 1001.0)
    assert not jl.is_takeable(_lease(), 1001.0)
    assert jl.is_takeable(_lease(), 2000.0)


def test_doc_ids_per_kind_and_scope() -> None:
    assert jl.doc_id(jl.LeaseKind.REPROCESS) == 'lease:reprocess'
    assert jl.doc_id(jl.LeaseKind.LABEL_OP, 'op_1') == 'lease:label_op:op_1'
    assert jl.doc_id('combine', 'cj_9') == 'lease:combine:cj_9'


def test_generations_start_at_one_and_increase() -> None:
    assert jl.next_generation(None) == 1
    assert jl.next_generation(_lease(generation=3)) == 4


def test_fence_window_is_ttl_minus_margin() -> None:
    assert jl.LEASE_TTL_S - jl.FENCE_MARGIN_S == 90.0
    assert jl.HEARTBEAT_TICK_S * 8 <= jl.LEASE_TTL_S
