"""The Z4 bars and the adoption rule: every boundary and every INCONCLUSIVE case."""

from __future__ import annotations

from typing import Any

from scripts.research.cluster_diag import z4_bars as zb


def _arm(u1: float, u2: float | None, open_u2: float | None, open_n: float) -> dict[str, Any]:
    def cell(v: float | None) -> dict[str, float] | None:
        return None if v is None else {'mean': v, 'min': v, 'max': v}

    return {
        'u1': cell(u1),
        'u2': cell(u2),
        'u2_open': cell(open_u2),
        'open_propagated': cell(open_n),
    }


def _results(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {r['id']: r['result'] for r in rows}


def test_a_clean_arm_passes_every_bar() -> None:
    rows = zb.arm_rows(_arm(0.3, 0.97, 0.95, 40), ivf_u1=0.8)
    assert _results(rows) == {'B-a': 'PASS', 'B-b': 'PASS', 'B-c': 'PASS'}
    assert zb.verdict(rows) == zb.PASS


def test_precision_bar_is_inclusive_at_0_95() -> None:
    assert _results(zb.arm_rows(_arm(0.3, 0.95, 0.95, 40), 0.8))['B-a'] == 'PASS'
    assert _results(zb.arm_rows(_arm(0.3, 0.9499, 0.95, 40), 0.8))['B-a'] == 'FAIL'


def test_effort_bar_is_half_of_ivf_and_inclusive() -> None:
    assert _results(zb.arm_rows(_arm(0.4, 0.97, 0.95, 40), 0.8))['B-b'] == 'PASS'
    assert _results(zb.arm_rows(_arm(0.4001, 0.97, 0.95, 40), 0.8))['B-b'] == 'FAIL'


def test_open_set_bar_is_inclusive_at_0_90_and_needs_30_propagated() -> None:
    assert _results(zb.arm_rows(_arm(0.3, 0.97, 0.90, 30), 0.8))['B-c'] == 'PASS'
    assert _results(zb.arm_rows(_arm(0.3, 0.97, 0.8999, 30), 0.8))['B-c'] == 'FAIL'
    short = zb.arm_rows(_arm(0.3, 0.97, 1.0, 29.9), 0.8)
    assert _results(short)['B-c'] == 'INCONCLUSIVE'
    assert zb.verdict(short) == zb.INCONCLUSIVE


def test_nothing_propagated_is_inconclusive_not_a_pass() -> None:
    rows = zb.arm_rows(_arm(1.0, None, None, 0), 0.8)
    assert _results(rows)['B-a'] == 'INCONCLUSIVE'
    assert _results(rows)['B-c'] == 'INCONCLUSIVE'


def test_a_failed_bar_outweighs_an_unreadable_one() -> None:
    rows = zb.arm_rows(_arm(0.3, 0.90, None, 0), 0.8)
    assert zb.verdict(rows) == zb.FAIL


def test_ivf_reference_is_the_cheaper_comparator() -> None:
    loop, gate = _arm(0.70, 0.99, None, 0), _arm(0.78, 0.99, None, 0)
    assert zb.ivf_reference(loop, gate) == 0.70
    assert zb.ivf_reference(gate, loop) == 0.70
    assert zb.ivf_reference({}, {}) is None


def test_adoption_takes_the_cheaper_passing_key_arm() -> None:
    v = {'key': zb.PASS, 'key_guard': zb.PASS, 'within_key': zb.PASS}
    assert zb.adoption(v, {'key': 0.30, 'key_guard': 0.35, 'within_key': 0.2})['adopt'] == 'key'
    assert zb.adoption(v, {'key': 0.40, 'key_guard': 0.35, 'within_key': 0.2})['adopt'] == (
        'key_guard'
    )
    # a tie goes to the simpler arm (no guard)
    assert zb.adoption(v, {'key': 0.3, 'key_guard': 0.3, 'within_key': 0.2})['adopt'] == 'key'


def test_adoption_falls_back_to_within_key_then_ivf() -> None:
    u1 = {'key': 0.3, 'key_guard': 0.3, 'within_key': 0.3}
    v = {'key': zb.FAIL, 'key_guard': zb.INCONCLUSIVE, 'within_key': zb.PASS}
    assert zb.adoption(v, u1)['adopt'] == 'within_key'
    v['within_key'] = zb.INCONCLUSIVE
    out = zb.adoption(v, u1)
    assert out['adopt'] == 'ivf'
    assert out['inconclusive_arms'] == ['key_guard', 'within_key']


def test_an_unreadable_key_arm_is_never_adopted() -> None:
    v = {'key': zb.INCONCLUSIVE, 'key_guard': zb.INCONCLUSIVE, 'within_key': zb.FAIL}
    assert zb.adoption(v, {'key': 0.1, 'key_guard': 0.1, 'within_key': 0.1})['adopt'] == 'ivf'


def test_overall_verdict() -> None:
    assert zb.overall({'key': 'FAIL', 'key_guard': 'FAIL', 'within_key': 'FAIL'}) == zb.FAIL
    assert zb.overall({'key': 'FAIL', 'key_guard': 'INCONCLUSIVE', 'within_key': 'FAIL'}) == (
        zb.INCONCLUSIVE
    )
    assert zb.overall({'key': 'FAIL', 'key_guard': 'PASS', 'within_key': 'FAIL'}) == zb.PASS
