"""Synthetic JPEGs, fused detector outputs and a mocked Triton pool for the region detector tests."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from _region_profile_fixture import EXAMPLE_LICENSE_PLATE_PROFILE as REFERENCE_LICENSE_PLATE_PROFILE
from PIL import Image


# =============================================================================
# Fixtures and helpers
# =============================================================================


def _make_jpeg(
    width: int = 320, height: int = 240, color: tuple[int, int, int] = (50, 80, 120)
) -> bytes:
    """Synthesize a small RGB JPEG suitable for the region detector input path."""
    buf = io.BytesIO()
    Image.new('RGB', (width, height), color).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def _make_raw_output(
    *,
    cx_norm: float,
    cy_norm: float,
    w_norm: float,
    h_norm: float,
    score: float,
) -> np.ndarray:
    """Build a synthetic fused ``[1, 300, 6]`` detector output.

    The box is given as center and size in fraction-of-network-input
    (so ``cx_norm=0.5`` is the center of the 640x640 letterbox) and written as
    ``x1, y1, x2, y2`` rows in network pixels, as the engine emits them.
    """
    arr = np.zeros((300, 6), dtype=np.float32)
    multiplier = float(REFERENCE_LICENSE_PLATE_PROFILE.input_size)

    def row(cx: float, cy: float, w: float, h: float, conf: float) -> list[float]:
        return [
            (cx - w / 2) * multiplier,
            (cy - h / 2) * multiplier,
            (cx + w / 2) * multiplier,
            (cy + h / 2) * multiplier,
            conf,
            0.0,
        ]

    # Row 0 = our planted detection at the requested score.
    arr[0] = row(cx_norm, cy_norm, w_norm, h_norm, score)
    # Row 1 = a low-score distractor so argmax has work to do.
    arr[1] = row(0.5, 0.5, 0.05, 0.02, 0.01)
    return arr[None, ...]


class _FakeInferResult:
    """Mimic the subset of ``InferResult`` the detector uses."""

    def __init__(self, output: np.ndarray) -> None:
        self._output = output

    def as_numpy(self, name: str) -> np.ndarray:  # noqa: ARG002 — single output
        return self._output


def _make_mock_pool(
    raw_output: np.ndarray | None = None,
    *,
    raises: BaseException | None = None,
    input_datatype: str = 'UINT8',
) -> MagicMock:
    """Build an :class:`AsyncTritonPool` mock that returns one canned output.

    ``input_datatype`` is what the mocked model declares for its ``images`` input.
    """
    pool = MagicMock()
    pool.get_model_input_datatype = AsyncMock(return_value=input_datatype)
    pool.get_model_output_shape = AsyncMock(return_value=(-1, 300, 6))
    if raises is not None:
        pool.infer = AsyncMock(side_effect=raises)
    else:
        pool.infer = AsyncMock(return_value=_FakeInferResult(raw_output))
    return pool
