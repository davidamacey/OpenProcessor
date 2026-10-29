"""FG-6 novelty scoring and the verdict-table assembly."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.fine_grained import fg6, summary_run
from scripts.research.fine_grained.data import Universe


def test_novelty_is_one_minus_mean_cosine_to_the_nearest_other_photo_examples() -> None:
    vec = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [1.0, 0.0]])
    groups = np.array(['a', 'b', 'c', 'a'])
    # row 3 (group a) vs examples 0..2 with k=2: row 0 is the same photo group, so it is skipped;
    # nearest two of {row1: 0.8, row2: 0.0} -> mean 0.4 -> novelty 0.6
    nov = fg6.novelty(vec, np.array([3]), np.array([0, 1, 2]), groups, k=2)
    assert nov[0] == pytest.approx(0.6)


def _universe(
    vec_by_model: dict[str, np.ndarray], make_of: dict[str, str], new: set[str]
) -> tuple[Universe, np.ndarray]:
    meta, make, model, half, vecs = [], [], [], [], []
    for m, v in vec_by_model.items():
        for i, x in enumerate(v):
            meta.append({'id': f'{m}_{i}', 'source': 'setc'})
            make.append(make_of[m])
            model.append(m)
            half.append('calibration' if i % 2 == 0 else 'heldout')
            vecs.append(x)
    arr = np.asarray(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    u = Universe(
        meta,
        np.array(half),
        np.array(make),
        np.array(model),
        np.array([''] * len(meta)),
        np.array([m in new for m in model]),
    )
    return u, arr


def test_a_new_model_far_from_the_known_ones_gets_auroc_one_and_the_flag_fires() -> None:
    rng = np.random.default_rng(0)

    def cloud(centre: list[float], n: int = 60) -> np.ndarray:
        return np.asarray(centre) + rng.normal(0, 0.03, (n, 3))

    u, vec = _universe(
        {'known': cloud([1, 0, 0]), 'fresh': cloud([0, 1, 0.2])},
        {'known': 'M', 'fresh': 'M'},
        {'fresh'},
    )
    res = fg6.run_fold(vec, u, 1)
    assert res['auroc_by_make']['M'] == pytest.approx(1.0)
    assert (
        res['pooled_flag_precision'] >= fg6.PRECISION_BAR
    )  # the lowest quantile meeting the bar is used
    assert res['per_make']['M']['new_recall'] == pytest.approx(1.0)


def test_a_new_model_that_looks_like_the_known_one_is_not_separable() -> None:
    rng = np.random.default_rng(1)
    same = np.array([1.0, 0.0, 0.0]) + rng.normal(0, 0.05, (120, 3))
    u, vec = _universe(
        {'known': same[:60], 'fresh': same[60:]}, {'known': 'M', 'fresh': 'M'}, {'fresh'}
    )
    assert fg6.run_fold(vec, u, 1)['auroc_by_make']['M'] == pytest.approx(0.5, abs=0.15)


def test_fg1_variants_pairs_each_variant_with_its_own_centre_and_needs_the_class_check() -> None:
    def fold(model: float, make: float) -> dict[str, dict[str, float]]:
        return {'model': {'P@10': model}, 'make': {'P@10': make}}

    screen = {
        'models': {
            'm__centre': {'fold1': fold(0.60, 0.80), 'fold2': fold(0.60, 0.80)},
            'm__letterbox': {'fold1': fold(0.66, 0.82), 'fold2': fold(0.66, 0.82)},
            'm__margin10': {'fold1': fold(0.61, 0.80), 'fold2': fold(0.61, 0.80)},
        }
    }
    no_class = summary_run.fg1_variants(screen, None)
    assert no_class['m__letterbox']['fold1']['verdict'] == 'INCONCLUSIVE'
    assert no_class['m__margin10']['fold1']['verdict'] == 'FAIL'
    cls = {'class_p10': {'m__letterbox': 0.70, 'm__centre': 0.70}}
    assert summary_run.fg1_variants(screen, cls)['m__letterbox']['fold1']['verdict'] == 'PASS'
    cls = {'class_p10': {'m__letterbox': 0.60, 'm__centre': 0.70}}
    assert summary_run.fg1_variants(screen, cls)['m__letterbox']['fold1']['verdict'] == 'FAIL'


def test_auroc_counts_ties_half() -> None:
    assert fg6.auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert fg6.auroc(np.array([1.0]), np.array([1.0])) == 0.5
    assert fg6.auroc(np.array([]), np.array([1.0])) is None


def _group(model: str, precision: float, make: float, r1: float) -> dict:
    test = {
        'model': {'bcubed_precision': precision},
        'make': {'bcubed_precision': make},
        'r1': r1,
    }
    folds = {
        'fold1': {'test': test},
        'fold2': {'test': {**test, 'model': {'bcubed_precision': precision - 0.08}}},
    }
    return {
        'model': model,
        'variant': 'centre',
        'methods': {'ahc': {'folds': folds, 'stability': {'permutation_ari': 1.0}}},
    }


def test_fg2_and_fg5_use_the_pe_core_r1_and_flag_a_fold2_miss(tmp_path) -> None:
    import json

    (tmp_path / 'group_a.json').write_text(json.dumps(_group('pe_core_l14_336', 0.92, 0.97, 0.50)))
    (tmp_path / 'group_b.json').write_text(json.dumps(_group('other', 0.93, 0.97, 0.40)))
    ch = {'other__centre': {'fold1': {'verdict': 'PASS'}}}
    out = summary_run.fg2_and_fg5(tmp_path, ch)
    assert out['pe_core_l14_336__centre']['FG-2']['verdict'] == 'PASS'
    assert out['other__centre']['FG-5'] == 'PASS'  # 0.40 <= 0.9 * 0.50
    assert out['other__centre']['reopen_fold2'] == ['model_bcubed_precision']  # 0.85 < 0.90 - 0.03
    ch = {'other__centre': {'fold1': {'verdict': 'FAIL'}}}
    assert summary_run.fg2_and_fg5(tmp_path, ch)['other__centre']['FG-5'] == 'FAIL'
