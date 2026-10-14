"""Synthetic JPEGs, raw YOLO outputs and a mocked Triton pool for the region detector tests."""

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
    n_anchors: int = 8400,
    pixel_space: bool = True,
    extra: tuple[float, float, float, float, float] | None = None,
) -> np.ndarray:
    """Build a synthetic ``[1, 5, N]`` raw YOLOv11-shaped output.

    ``cx_norm`` etc are in fraction-of-network-input (so ``cx_norm=0.5``
    is the center of the 640x640 letterbox). When ``pixel_space=True``
    we multiply by 640 to match the actual model export. When False we
    leave normalized to exercise the decoder's "looks normalized?"
    fallback branch.
    """
    arr = np.zeros((5, n_anchors), dtype=np.float32)
    multiplier = float(REFERENCE_LICENSE_PLATE_PROFILE.input_size) if pixel_space else 1.0
    # Anchor 0 = our planted detection at the requested score.
    arr[0, 0] = cx_norm * multiplier
    arr[1, 0] = cy_norm * multiplier
    arr[2, 0] = w_norm * multiplier
    arr[3, 0] = h_norm * multiplier
    arr[4, 0] = score
    # Anchor 1 = a low-score distractor so argmax has work to do.
    if n_anchors > 1:
        arr[0, 1] = 0.5 * multiplier
        arr[1, 1] = 0.5 * multiplier
        arr[2, 1] = 0.05 * multiplier
        arr[3, 1] = 0.02 * multiplier
        arr[4, 1] = 0.01
    if extra is not None and n_anchors > 2:
        arr[0, 2] = extra[0] * multiplier
        arr[1, 2] = extra[1] * multiplier
        arr[2, 2] = extra[2] * multiplier
        arr[3, 2] = extra[3] * multiplier
        arr[4, 2] = extra[4]
    return arr[None, ...]  # add batch dim → [1, 5, N]


class _FakeInferResult:
    """Mimic the subset of ``InferResult`` the detector uses."""

    def __init__(self, output: np.ndarray) -> None:
        self._output = output

    def as_numpy(self, name: str) -> np.ndarray:  # noqa: ARG002 — single output
        return self._output


def _make_mock_pool(
    raw_output: np.ndarray | None = None, *, raises: BaseException | None = None
) -> MagicMock:
    """Build an :class:`AsyncTritonPool` mock that returns one canned output."""
    pool = MagicMock()
    if raises is not None:
        pool.infer = AsyncMock(side_effect=raises)
    else:
        pool.infer = AsyncMock(return_value=_FakeInferResult(raw_output))
    return pool
