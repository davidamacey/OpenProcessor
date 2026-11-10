"""Wire model for ``POST /cluster/umap/rebuild``. Leaf module."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class UmapRebuildResponse(BaseModel):
    """``no_residuals`` carries only ``status``, ``n_residuals`` and ``refit``; the backend keys
    appear on ``success`` (served with ``response_model_exclude_unset``)."""

    status: Literal['success', 'no_residuals']
    n_residuals: int
    refit: bool
    backend: str | None = None
    backend_detail: str | None = None
