"""Synthetic-geometry coverage for ``src/services/detection/region_lean.py``
(plan Wave 5 W5.b — a zero-coverage pure cv2/numpy leaf).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.services.detection.region_lean import (
    LeanEstimate,
    TextLine,
    estimate_from_lines,
    estimate_lean,
    lean_from_lines,
    lines_from_polys,
    longest_edge_angle,
)


def _quad_at_angle(angle_deg: float, *, length: float = 100.0, height: float = 20.0) -> np.ndarray:
    """A rectangular text-line quad whose longest edge is tilted
    ``angle_deg`` from horizontal, centered at the origin."""
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad) * length / 2, math.sin(rad) * length / 2
    # Perpendicular offset for the short edge (line height).
    pdx, pdy = -math.sin(rad) * height / 2, math.cos(rad) * height / 2
    top_left = (-dx + pdx, -dy + pdy)
    top_right = (dx + pdx, dy + pdy)
    bottom_right = (dx - pdx, dy - pdy)
    bottom_left = (-dx - pdx, -dy - pdy)
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


@pytest.mark.parametrize('angle', [0.0, 15.0, -15.0, 45.0, -45.0])
def test_longest_edge_angle_recovers_the_synthetic_tilt(angle: float) -> None:
    quad = _quad_at_angle(angle)
    recovered = longest_edge_angle(quad)
    # +/-45 wrap to the same value at the boundary; otherwise exact.
    assert recovered == pytest.approx(angle, abs=1e-3) or recovered == pytest.approx(
        -angle, abs=1e-3
    )


def test_longest_edge_angle_normalizes_into_the_bounded_range() -> None:
    for angle in (0.0, 15.0, -15.0, 44.9, -44.9):
        recovered = longest_edge_angle(_quad_at_angle(angle))
        assert -45 < recovered <= 45


def test_longest_edge_angle_degenerate_collinear_quad_does_not_raise() -> None:
    # All four points on one horizontal line — zero-height quad.
    quad = np.array([[0, 0], [10, 0], [10, 0], [0, 0]], dtype=np.float32)
    angle = longest_edge_angle(quad)
    assert angle == pytest.approx(0.0, abs=1e-3)


def test_longest_edge_angle_degenerate_single_point_does_not_raise() -> None:
    quad = np.array([[5, 5], [5, 5], [5, 5], [5, 5]], dtype=np.float32)
    # A zero-length "longest edge" still has a well-defined atan2(0, 0) = 0.
    assert longest_edge_angle(quad) == pytest.approx(0.0, abs=1e-3)


def test_lines_from_polys_handles_none_and_empty() -> None:
    assert lines_from_polys(None) == []
    assert lines_from_polys([]) == []


def test_lines_from_polys_builds_one_textline_per_poly() -> None:
    polys = [_quad_at_angle(0.0, length=50), _quad_at_angle(10.0, length=100)]
    lines = lines_from_polys(polys)
    assert len(lines) == 2
    assert all(isinstance(ln, TextLine) for ln in lines)
    # Longer polygon (length=100) has the larger recorded length.
    assert lines[1].length > lines[0].length


def test_lean_from_lines_empty_returns_none() -> None:
    angle, spread, n = lean_from_lines([])
    assert angle is None
    assert spread == 0.0
    assert n == 0


def test_lean_from_lines_frontal_agreement_low_spread() -> None:
    # 3 lines all tilted the same 10 degrees; keep_frac=0.6 of 3 rounds to
    # 2 kept -> zero spread (they all agree), median 10.
    lines = [TextLine(10.0, 100.0, _quad_at_angle(10.0)) for _ in range(3)]
    angle, spread, n = lean_from_lines(lines)
    assert angle == pytest.approx(10.0)
    assert spread == pytest.approx(0.0)
    assert n == 2


def test_lean_from_lines_keep_frac_1_includes_every_line() -> None:
    lines = [TextLine(10.0, 100.0, _quad_at_angle(10.0)) for _ in range(4)]
    _angle, _spread, n = lean_from_lines(lines, keep_frac=1.0)
    assert n == 4


def test_estimate_from_lines_frontal_flag() -> None:
    frontal_lines = [TextLine(2.0, 100.0, _quad_at_angle(2.0)) for _ in range(3)]
    est = estimate_from_lines(frontal_lines)
    assert isinstance(est, LeanEstimate)
    assert est.frontal is True
    assert est.angle_deg == pytest.approx(2.0)

    # Wide disagreement among the kept (equal-length, so order-preserved)
    # lines -> oblique, not frontal. 4 lines, keep_frac=0.6 keeps the
    # first 2 (equal length -> stable order), which disagree by 40 deg.
    oblique_lines = [
        TextLine(0.0, 100.0, _quad_at_angle(0.0)),
        TextLine(40.0, 100.0, _quad_at_angle(40.0)),
        TextLine(0.0, 100.0, _quad_at_angle(0.0)),
        TextLine(40.0, 100.0, _quad_at_angle(40.0)),
    ]
    est2 = estimate_from_lines(oblique_lines)
    assert est2.spread_deg > 6.0
    assert est2.frontal is False


def test_estimate_lean_end_to_end_from_polys() -> None:
    # 3 polys, same angle, decreasing length; keep_frac=0.6 of 3 keeps
    # the 2 longest.
    polys = [
        _quad_at_angle(15.0, length=100),
        _quad_at_angle(15.0, length=90),
        _quad_at_angle(15.0, length=80),
    ]
    est = estimate_lean(polys)
    assert est.angle_deg == pytest.approx(15.0, abs=0.5)
    assert est.text_lines == 2


def test_estimate_lean_no_text_returns_none_angle() -> None:
    est = estimate_lean(None)
    assert est.angle_deg is None
    assert est.text_lines == 0
    assert est.frontal is False
