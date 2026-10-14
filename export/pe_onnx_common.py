"""Helpers shared by the PE text and image encoder exporters."""

from __future__ import annotations

from typing import Any


def _is_dynamic(dim: Any) -> bool:
    """Whether an ONNX dimension is symbolic (i.e. batchable)."""
    return isinstance(dim, str) or dim is None or (isinstance(dim, int) and dim < 1)
