"""Exact one-sided bounds behind the auto-label loop's gate (design section 6)."""

from __future__ import annotations

import math

import pytest

from src.domain.loop_stats import homogeneity_lower_bound, required_sample, wilson_lower_bound


LARGE_N = 10**6

# Design section 6, first table: alpha 0.05, large node, dissent 0/1/2.
ONE_STAGE = {
    0.80: (14, 22, 30),
    0.85: (19, 30, 40),
    0.90: (29, 46, 61),
    0.95: (59, 93, 124),
}

# Design section 6, second table: alpha_stage 0.025, finite N, dissent 0/1.
NODE_SIZES = (30, 60, 150, 500, 2000, 8000)
TWO_STAGE = {
    (0.80, 0): (11, 14, 16, 17, 17, 17),
    (0.80, 1): (17, 20, 23, 25, 26, 26),
    (0.90, 0): (18, 24, 30, 34, 35, 35),
    (0.90, 1): (24, 34, 44, 51, 53, 54),
    (0.95, 0): (25, 36, 55, 65, 70, 72),
    (0.95, 1): (30, 48, 78, 97, 106, 109),
}


@pytest.mark.parametrize(('theta', 'expected'), sorted(ONE_STAGE.items()))
def test_large_node_table(theta: float, expected: tuple[int, ...]) -> None:
    got = tuple(required_sample(LARGE_N, theta, 0.05, d) for d in range(3))
    assert got == expected


@pytest.mark.parametrize(('theta', 'dissent'), sorted(TWO_STAGE))
def test_two_stage_table(theta: float, dissent: int) -> None:
    got = tuple(required_sample(n, theta, 0.025, dissent) for n in NODE_SIZES)
    assert got == TWO_STAGE[theta, dissent]


def test_large_node_matches_the_binomial_limit() -> None:
    # theta**n <= alpha with no dissent: n >= ln(alpha) / ln(theta)
    for theta in (0.8, 0.9, 0.95):
        assert required_sample(LARGE_N, theta, 0.05, 0) == math.ceil(
            math.log(0.05) / math.log(theta)
        )


@pytest.mark.parametrize('n_members', [1, 2, 3])
def test_small_node_is_asked_whole(n_members: int) -> None:
    assert required_sample(n_members, 0.8, 0.025, 0) == n_members


def test_never_exceeds_the_node() -> None:
    for n_members in range(1, 80):
        assert required_sample(n_members, 0.95, 0.001, 3) <= n_members


def test_monotone_in_theta_dissent_and_alpha() -> None:
    thetas = (0.7, 0.8, 0.9, 0.95, 0.99)
    for d in range(4):
        sizes = [required_sample(2000, t, 0.05, d) for t in thetas]
        assert sizes == sorted(sizes)
    for theta in thetas:
        by_dissent = [required_sample(2000, theta, 0.05, d) for d in range(4)]
        assert by_dissent == sorted(by_dissent)
        assert len(set(by_dissent)) == 4
        by_alpha = [required_sample(2000, theta, a, 1) for a in (0.1, 0.05, 0.01, 0.001)]
        assert by_alpha == sorted(by_alpha)


def test_required_sample_rejects_bad_arguments() -> None:
    for kwargs in (
        {'n_members': 0, 'theta': 0.9, 'alpha': 0.05, 'dissent': 0},
        {'n_members': 10, 'theta': 1.0, 'alpha': 0.05, 'dissent': 0},
        {'n_members': 10, 'theta': 0.0, 'alpha': 0.05, 'dissent': 0},
        {'n_members': 10, 'theta': 0.9, 'alpha': 0.0, 'dissent': 0},
        {'n_members': 10, 'theta': 0.9, 'alpha': 1.0, 'dissent': 0},
        {'n_members': 10, 'theta': 0.9, 'alpha': 0.05, 'dissent': -1},
    ):
        with pytest.raises(ValueError, match='must'):
            required_sample(**kwargs)


def test_lower_bound_agrees_with_the_gate() -> None:
    # The gate passes exactly when the bound reaches theta.
    for n_members, theta, dissent in ((60, 0.9, 0), (150, 0.9, 1), (500, 0.8, 1)):
        need = required_sample(n_members, theta, 0.05, dissent)
        passes = homogeneity_lower_bound(need - dissent, need, n_members, 0.05)
        fails = homogeneity_lower_bound(need - 1 - dissent, need - 1, n_members, 0.05)
        assert passes >= theta
        assert fails < theta


def test_lower_bound_whole_node_is_exact() -> None:
    assert homogeneity_lower_bound(30, 30, 30, 0.05) == 1.0
    assert homogeneity_lower_bound(27, 30, 30, 0.05) == 27 / 30


def test_lower_bound_is_never_below_what_was_seen() -> None:
    bound = homogeneity_lower_bound(10, 10, 1000, 0.05)
    assert 0.0 < bound < 1.0
    assert bound >= 10 / 1000


def test_lower_bound_rises_with_evidence() -> None:
    low = homogeneity_lower_bound(10, 10, 1000, 0.05)
    high = homogeneity_lower_bound(50, 50, 1000, 0.05)
    assert high > low


def test_lower_bound_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match='must'):
        homogeneity_lower_bound(11, 10, 100, 0.05)
    with pytest.raises(ValueError, match='must'):
        homogeneity_lower_bound(5, 101, 100, 0.05)


@pytest.mark.parametrize(
    ('agreed', 'total', 'expected'),
    [(95, 100, 0.90), (90, 100, 0.84), (48, 50, 0.886)],
)
def test_wilson_examples_of_the_design(agreed: int, total: int, expected: float) -> None:
    assert wilson_lower_bound(agreed, total, 1.645) == pytest.approx(expected, abs=0.001)


def test_wilson_edges() -> None:
    assert wilson_lower_bound(0, 0, 1.645) == 0.0
    assert wilson_lower_bound(0, 40, 1.645) == 0.0
    assert 0.9 < wilson_lower_bound(40, 40, 1.645) < 1.0
    with pytest.raises(ValueError, match='must'):
        wilson_lower_bound(5, 4, 1.645)
