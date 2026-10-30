"""The use-case bars U-a to U-d: boundaries and the INCONCLUSIVE rules."""

from __future__ import annotations

from typing import Any

from scripts.research.cluster_diag import u_bars as ub


def _arm(u1: float, u2: float | None, open_u2: float | None = None, open_n: float = 0.0) -> Any:
    def cell(v: float | None) -> Any:
        return None if v is None else {'mean': v, 'min': v, 'max': v}

    return {
        'u1': cell(u1),
        'u2': cell(u2),
        'u2_open': cell(open_u2),
        'open_propagated': cell(open_n),
    }


def _results(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {r['id']: r['result'] for r in rows}


def test_all_pass() -> None:
    rows = ub.u_bars(_arm(0.2, 0.96, 0.92, 100), _arm(0.5, 0.97), _arm(0.3, 0.96))
    assert _results(rows) == {'U-a': 'PASS', 'U-b': 'PASS', 'U-c': 'PASS', 'U-d': 'PASS'}
    assert ub.overall(rows) == 'PASS'


def test_the_boundaries_are_inclusive_and_just_past_them_fail() -> None:
    # U-a 0.95; U-b 0.5 x 0.6 = 0.30; U-c 0.9 x 0.4 = 0.36; U-d 0.90
    on = ub.u_bars(_arm(0.30, 0.95, 0.90, 50), _arm(0.6, 0.97), _arm(0.4, 0.97))
    assert _results(on)['U-a'] == 'PASS'
    assert _results(on)['U-b'] == 'PASS'
    assert _results(on)['U-d'] == 'PASS'
    # 0.37 is over both U-b's 0.30 and U-c's 0.36
    off = ub.u_bars(_arm(0.37, 0.949, 0.899, 50), _arm(0.6, 0.97), _arm(0.4, 0.97))
    assert _results(off) == {'U-a': 'FAIL', 'U-b': 'FAIL', 'U-c': 'FAIL', 'U-d': 'FAIL'}
    assert ub.overall(off) == 'FAIL'


def test_u_b_fails_when_ivf_is_cheap() -> None:
    rows = ub.u_bars(_arm(0.4, 0.97, 0.95, 80), _arm(0.5, 0.97), _arm(0.9, 0.97))
    assert _results(rows)['U-b'] == 'FAIL'  # 0.4 > 0.5 x 0.5


def test_u_c_is_inconclusive_when_the_key_arm_is_below_the_floor() -> None:
    rows = ub.u_bars(_arm(0.2, 0.97, 0.95, 80), _arm(0.5, 0.97), _arm(0.3, 0.90))
    assert _results(rows)['U-c'] == 'INCONCLUSIVE'
    assert ub.overall(rows) == 'INCONCLUSIVE'


def test_u_c_fails_when_the_method_itself_is_below_the_floor() -> None:
    rows = ub.u_bars(_arm(0.2, 0.90, 0.95, 80), _arm(0.5, 0.97), _arm(0.3, 0.97))
    assert _results(rows)['U-c'] == 'FAIL'


def test_u_d_needs_thirty_propagated_open_set_items() -> None:
    rows = ub.u_bars(_arm(0.2, 0.97, 1.0, 29.9), _arm(0.5, 0.97), _arm(0.3, 0.97))
    assert _results(rows)['U-d'] == 'INCONCLUSIVE'
    rows = ub.u_bars(_arm(0.2, 0.97, 1.0, 30.0), _arm(0.5, 0.97), _arm(0.3, 0.97))
    assert _results(rows)['U-d'] == 'PASS'


def test_nothing_propagated_is_inconclusive_not_a_pass() -> None:
    rows = ub.u_bars(_arm(1.0, None), _arm(0.5, 0.97), _arm(0.3, 0.97))
    assert _results(rows)['U-a'] == 'INCONCLUSIVE'
    assert _results(rows)['U-b'] == 'FAIL'  # 1.0 > 0.25


def test_the_simpler_winner_is_the_cheaper_arm_at_the_floor() -> None:
    arms = {'ivf': _arm(0.5, 0.97), 'key_only': _arm(0.3, 0.96)}
    assert ub.simpler_winner(arms) == 'key_only'
    arms = {'ivf': _arm(0.2, 0.97), 'key_only': _arm(0.3, 0.96)}
    assert ub.simpler_winner(arms) == 'ivf'
    arms = {'ivf': _arm(0.2, 0.94), 'key_only': _arm(0.3, 0.96)}
    assert ub.simpler_winner(arms) == 'key_only'  # the cheaper one misses the floor
    arms = {'ivf': _arm(0.2, 0.94), 'key_only': _arm(0.3, 0.94)}
    assert ub.simpler_winner(arms) is None
