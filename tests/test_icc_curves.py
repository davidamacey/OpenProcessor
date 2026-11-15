"""The tone-curve reader (``src/utils/icc_curves.py``) against what the profiles define.

Reference values: the sRGB curve of IEC 61966-2.1 (Display P3 and the sRGB profiles use it) and the
gamma 563/256 of Adobe RGB (1998), which the compatible v4 profile writes with a linear toe.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from src.utils.icc_curves import tone_curves


ICC = Path(__file__).resolve().parent / 'fixtures' / 'icc'
X = np.arange(256) / 255.0


def srgb_eotf(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


@pytest.mark.parametrize('name', ['DisplayP3-v4.icc', 'colord-sRGB.icc'])
def test_an_srgb_curve_profile_reads_as_the_srgb_curve(name: str) -> None:
    curves = tone_curves((ICC / name).read_bytes())
    assert curves is not None
    for c in curves:
        assert np.abs(c(X) - srgb_eotf(X)).max() < 2e-4


def test_adobe_compatible_profile_reads_as_gamma_563_256_with_a_linear_toe() -> None:
    """Its curve is the parametric type 3: ``x ** g`` from ``d`` up, the straight line ``c * x`` below."""
    curves = tone_curves((ICC / 'AdobeCompat-v4.icc').read_bytes())
    assert curves is not None
    upper = X >= 0.1
    for c in curves:
        assert np.abs(c(X)[upper] - X[upper] ** (563 / 256)).max() < 2e-4
        assert np.allclose(c(X[:3]), X[:3] / 32, atol=1e-6)  # the toe: c = 1/32


def _profile_with_curve(tag_data: bytes) -> bytes:
    """A minimal byte string with the three curve tags all pointing at ``tag_data``."""
    table = 128 + 4 + 3 * 12
    head = bytes(128) + struct.pack('>I', 3)
    for sig in (b'rTRC', b'gTRC', b'bTRC'):
        head += sig + struct.pack('>II', table, len(tag_data))
    return head + tag_data


def test_a_table_curve_is_interpolated_linearly() -> None:
    table = struct.pack('>IIIHH', 0x63757276, 0, 2, 0, 65535)  # 'curv', two entries: identity
    curves = tone_curves(_profile_with_curve(table))
    assert curves is not None
    assert np.allclose(curves[0](X), X)


def test_an_empty_curve_is_the_identity() -> None:
    curves = tone_curves(_profile_with_curve(struct.pack('>III', 0x63757276, 0, 0)))
    assert curves is not None
    assert np.allclose(curves[1](X), X)


def test_a_type_3_parametric_curve_follows_its_function() -> None:
    def s15(v: float) -> int:
        return round(v * 65536)

    # IEC 61966-2.1: g 2.4, a 1/1.055, b 0.055/1.055, c 1/12.92, d 0.04045
    data = struct.pack('>IIHH', 0x70617261, 0, 3, 0) + struct.pack(
        '>5i', s15(2.4), s15(1 / 1.055), s15(0.055 / 1.055), s15(1 / 12.92), s15(0.04045)
    )
    curves = tone_curves(_profile_with_curve(data))
    assert curves is not None
    assert np.abs(curves[2](X) - srgb_eotf(X)).max() < 2e-4


@pytest.mark.parametrize(
    'icc',
    [
        b'',
        bytes(130),
        bytes(128) + struct.pack('>I', 1 << 30),
        _profile_with_curve(struct.pack('>III', 0x63757276, 0, 1000)),  # claims 1000 entries
        _profile_with_curve(struct.pack('>IIHH', 0x70617261, 0, 9, 0)),  # unknown function type
        _profile_with_curve(b'text' + bytes(8)),
        _profile_with_curve(b'short'),
    ],
)
def test_bytes_that_are_not_curves_give_none(icc: bytes) -> None:
    assert tone_curves(icc) is None


def test_a_tag_that_leaves_the_file_gives_none() -> None:
    icc = bytearray(_profile_with_curve(struct.pack('>III', 0x63757276, 0, 0)))
    struct.pack_into('>II', icc, 136, 10_000, 12)  # first entry's offset, size
    assert tone_curves(bytes(icc)) is None
