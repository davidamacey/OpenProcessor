"""The tone curves of an ICC matrix/TRC profile, read from its tag table.

A matrix/TRC RGB profile (every camera, phone and monitor profile in common use) is three
one-dimensional curves (``rTRC``, ``gTRC``, ``bTRC``) and a 3 x 3 matrix (``rXYZ``, ``gXYZ``,
``bXYZ``). Pillow exposes the matrix (``ImageCmsProfile.profile.red_colorant`` and the other two)
but not the curves, so they are read here as the ICC specification lays them out (ICC.1:2022,
https://www.color.org/specification/ICC.1-2022-05.pdf): the tag table (section 7.3: a tag count at
byte 128, then ``signature, offset, size`` of 4 bytes each), the ``curveType`` (section 10.6) and
the ``parametricCurveType`` (section 10.18, function types 0 to 4).

Everything is read from untrusted bytes: any tag, size or type that is not exactly as the
specification allows gives ``None`` and the caller falls back to the transform-based test.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import cast

import numpy as np


Curve = Callable[[np.ndarray], np.ndarray]

_TAG_TABLE = 128
_CURVE_TAGS = (b'rTRC', b'gTRC', b'bTRC')
#: ``parametricCurveType`` parameter counts by function type (ICC.1:2022 table 65).
_PARA_PARAMS = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}


def _s15(raw: bytes) -> float:
    (value,) = struct.unpack('>i', raw)
    return float(value) / 65536.0


def _tag_data(icc: bytes, signature: bytes) -> bytes | None:
    """The bytes of one tag, or ``None`` when it is absent or its entry leaves the file."""
    if len(icc) < _TAG_TABLE + 4:
        return None
    (count,) = struct.unpack('>I', icc[_TAG_TABLE : _TAG_TABLE + 4])
    if count > (len(icc) - _TAG_TABLE - 4) // 12:
        return None
    for i in range(count):
        at = _TAG_TABLE + 4 + 12 * i
        if icc[at : at + 4] == signature:
            offset, size = struct.unpack('>II', icc[at + 4 : at + 12])
            if offset + size > len(icc):
                return None
            return icc[offset : offset + size]
    return None


def _parametric(data: bytes) -> Curve | None:
    if len(data) < 12:
        return None
    (kind,) = struct.unpack('>H', data[8:10])
    n = _PARA_PARAMS.get(kind)
    if n is None or len(data) < 12 + 4 * n:
        return None
    p = [_s15(data[12 + 4 * k : 16 + 4 * k]) for k in range(n)]
    g = p[0]

    def curve(x: np.ndarray) -> np.ndarray:
        if kind == 0:
            return x**g
        _, a, b, *rest = p
        if a == 0:
            return np.full_like(x, np.nan)
        # ``base`` is only taken where its argument is not negative (a fractional power of a
        # negative number is undefined); the branch below picks the segment for every x
        base = np.power(np.maximum(a * x + b, 0.0), g)
        if kind == 1:
            return np.where(x >= -b / a, base, 0.0)
        if kind == 2:
            c = rest[0]
            return np.where(x >= -b / a, base + c, c)
        c, d = rest[0], rest[1]
        if kind == 3:
            return np.where(x >= d, base, c * x)
        e, f = rest[2], rest[3]
        return np.where(x >= d, base + e, c * x + f)

    return curve


def _curve(data: bytes) -> Curve | None:
    if len(data) < 12:
        return None
    if data[:4] == b'para':
        return _parametric(data)
    if data[:4] != b'curv':
        return None
    (n,) = struct.unpack('>I', data[8:12])
    if len(data) < 12 + 2 * n:
        return None
    if n == 0:
        return lambda x: x.copy()
    if n == 1:
        gamma = struct.unpack('>H', data[12:14])[0] / 256.0  # u8Fixed8Number
        return lambda x: x**gamma
    table = np.frombuffer(data[12 : 12 + 2 * n], '>u2').astype(np.float64) / 65535.0
    grid = np.linspace(0.0, 1.0, n)
    return lambda x: np.interp(x, grid, table)


def tone_curves(icc: bytes) -> tuple[Curve, Curve, Curve] | None:
    """The red, green and blue curves (input in ``[0, 1]`` to linear light), or ``None`` when the
    profile does not hold the three in a form this module reads."""
    found = [_tag_data(icc, sig) for sig in _CURVE_TAGS]
    curves = [None if d is None else _curve(d) for d in found]
    if any(c is None for c in curves):
        return None
    return cast('tuple[Curve, Curve, Curve]', tuple(curves))
