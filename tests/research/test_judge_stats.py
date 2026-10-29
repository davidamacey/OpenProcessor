"""judge_stats: Wilson and cluster-bootstrap intervals, Cohen's and Fleiss' kappa."""

from __future__ import annotations

import math

import pytest

from scripts.research import judge_stats as st


def test_wilson_matches_a_hand_computed_interval() -> None:
    lo, hi = st.wilson_interval(45, 50)
    assert lo == pytest.approx(0.7864, abs=1e-3)
    assert hi == pytest.approx(0.9565, abs=1e-3)


def test_wilson_edges_stay_inside_zero_one_and_empty_is_uninformative() -> None:
    lo, hi = st.wilson_interval(0, 20)
    assert lo == 0.0
    assert 0.1 < hi < 0.2
    lo, hi = st.wilson_interval(20, 20)
    assert hi == 1.0
    assert 0.8 < lo < 0.9
    assert st.wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match='successes'):
        st.wilson_interval(5, 3)


def test_cohens_kappa_known_values() -> None:
    assert st.cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert st.cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    a = ['y'] * 20 + ['n'] * 5 + ['y'] * 10 + ['n'] * 15
    b = ['y'] * 20 + ['y'] * 5 + ['n'] * 10 + ['n'] * 15
    assert st.cohens_kappa(a, b) == pytest.approx(0.4, abs=1e-9)


def test_cohens_kappa_is_undefined_without_variation_and_checks_lengths() -> None:
    assert st.cohens_kappa([1, 1, 1], [1, 1, 1]) is None
    with pytest.raises(ValueError, match='same length'):
        st.cohens_kappa([1], [1, 0])


def test_fleiss_kappa_perfect_and_chance() -> None:
    perfect = [['a', 'a', 'a'], ['b', 'b', 'b'], ['a', 'a', 'a'], ['b', 'b', 'b']]
    assert st.fleiss_kappa(perfect) == 1.0
    table = (
        (0, 0, 0, 0, 14),
        (0, 2, 6, 4, 2),
        (0, 0, 3, 5, 6),
        (0, 3, 9, 2, 0),
        (2, 2, 8, 1, 1),
        (7, 7, 0, 0, 0),
        (3, 2, 6, 3, 0),
        (2, 5, 3, 2, 2),
        (6, 5, 2, 1, 0),
        (0, 2, 2, 3, 7),
    )
    ratings = [[cat for cat, n in enumerate(c) for _ in range(n)] for c in table]
    assert st.fleiss_kappa(ratings) == pytest.approx(0.210, abs=2e-3)  # the textbook example


def test_fleiss_needs_equal_rater_counts() -> None:
    with pytest.raises(ValueError, match='raters'):
        st.fleiss_kappa([['a', 'b'], ['a']])


def test_cluster_bootstrap_resamples_sheets_not_cells() -> None:
    sheets = {f's{k}': [1.0] * 10 for k in range(10)}
    sheets.update({f't{k}': [0.0] * 10 for k in range(10)})
    lo, hi = st.cluster_bootstrap(sheets, seed=1)
    assert lo < 0.5 < hi
    assert lo == st.cluster_bootstrap(sheets, seed=1)[0]  # seeded
    # cells inside a sheet are perfectly correlated: the interval is that of 20 sheets, not 200
    wilson_lo, wilson_hi = st.wilson_interval(100, 200)
    assert (hi - lo) > (wilson_hi - wilson_lo)


def test_cluster_bootstrap_of_constant_data_is_degenerate_and_empty_is_none() -> None:
    lo, hi = st.cluster_bootstrap({'a': [1.0, 1.0], 'b': [1.0]}, seed=0)
    assert (lo, hi) == (1.0, 1.0)
    assert st.cluster_bootstrap({}, seed=0) is None


def test_binary_scores_flag_precision_recall_and_false_same_rate() -> None:
    truth = [True, True, True, False, False, False, False, False]
    pred = [True, True, False, True, False, False, False, False]
    s = st.binary_scores(truth, pred)
    assert s['n'] == 8
    assert s['accuracy'] == pytest.approx(6 / 8)
    assert s['precision'] == pytest.approx(2 / 3)
    assert s['recall'] == pytest.approx(2 / 3)
    assert s['false_negative_rate'] == pytest.approx(1 / 3)
    assert s['false_positive_rate'] == pytest.approx(1 / 5)
    assert math.isclose(s['kappa'], st.cohens_kappa(truth, pred))  # type: ignore[arg-type]
