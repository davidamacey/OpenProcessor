"""Five-point alignment geometry of the ``faces`` branch (``frame_engine/faces/alignment.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from frame_engine.faces import alignment as fa
from src.utils.face_align import ARCFACE_REF, _umeyama


def test_template_is_insightfaces_arcface_dst() -> None:
    np.testing.assert_array_equal(fa.ARCFACE_DST.astype(np.float32), ARCFACE_REF)


def test_estimate_agrees_with_todays_umeyama_on_real_like_landmarks() -> None:
    """Today's ``_umeyama`` (no rank branch) and scikit-image's solve agree on full-rank input;
    catches a transposed covariance, a scale from the wrong variance or a wrong translation."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        s, theta = rng.uniform(0.3, 8.0), rng.uniform(-1.2, 1.2)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        lmk = (fa.ARCFACE_DST @ rot.T) * s + rng.uniform(0, 4000, 2) + rng.normal(0, 2.0, (5, 2))
        np.testing.assert_allclose(fa.estimate(lmk), _umeyama(lmk, fa.ARCFACE_DST), atol=1e-9)


def test_estimate_maps_a_scaled_template_onto_the_template() -> None:
    lmk = fa.ARCFACE_DST * 4.0 + [1000.0, 500.0]
    m = fa.estimate(lmk)
    np.testing.assert_allclose(m, [[0.25, 0, -250.0], [0, 0.25, -125.0]], atol=1e-9)
    inv = fa.inverse(m)
    np.testing.assert_allclose(inv, [[4.0, 0, 1000.0], [0, 4.0, 500.0]], atol=1e-9)
    assert fa.samples(inv) == 4


@pytest.mark.parametrize(
    ('scale', 'taps'), [(0.5, 1), (1.0, 1), (1.3, 2), (7.9, 8), (40.0, fa.MAX_SAMPLES)]
)
def test_taps_cover_the_footprint(scale: float, taps: int) -> None:
    """One output pixel spans ``scale`` source pixels; the area filter needs that many taps per
    axis (never fewer than one, never more than the cap). Catches an inverted scale."""
    lmk = fa.ARCFACE_DST * scale
    _, t, ok = fa.warp_params(lmk[None])
    assert ok.tolist() == [True]
    assert t.tolist() == [taps]


def test_degenerate_landmarks_are_not_embedded() -> None:
    lmk = np.stack([fa.ARCFACE_DST * 2, np.full((5, 2), 7.0)])
    inv, taps, ok = fa.warp_params(lmk)
    assert ok.tolist() == [True, False]
    assert inv.dtype == np.float32
    assert taps.dtype == np.int32
