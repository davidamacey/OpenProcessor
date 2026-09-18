"""Clustering-method interface for the residual pool.

Each method (AHC, HDBSCAN, future additions) exposes the same async
``fit_predict`` so :py:func:`src.services.curation.clustering.orchestrator.cluster_residuals`
can dispatch through a registry rather than hard-coding the algorithm.

Shape contract:

* Input — an ``(n, d)`` float32 array of L2-normalized PE embeddings.
* Output — a :class:`ClusterResult` whose ``labels`` is an ``int64``
  array of length ``n``. ``-1`` means "noise" / unassigned (HDBSCAN
  emits these; AHC does not but the field is reserved). Non-negative
  labels are dense (``0..K-1``); ``cluster_residuals`` adds the
  ``RESIDUAL_CLUSTER_ID_OFFSET`` after stable-sorting them by size.

Methods are responsible for:

* Picking a backend (GPU/CPU) given the :class:`BackendInfo` probe
  ``cluster_residuals`` already collected — they should not call
  ``detect_cluster_backend`` themselves.
* Wrapping any blocking compute in ``asyncio.to_thread`` so the worker's
  heartbeat coroutine continues to fire.
* Calling ``progress.raise_if_cancelled()`` at coarse boundaries
  (between phases) so a cancel sentinel takes effect without waiting
  for the whole fit to finish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable


if TYPE_CHECKING:
    import numpy as np

    from src.services.curation.clustering.backend import BackendInfo


@dataclass(frozen=True)
class ClusterResult:
    """One fit's output. Surfaced verbatim in the auto-label job summary."""

    labels: np.ndarray
    """int64; ``-1`` = noise, ``>=0`` = cluster index (dense, ``0..K-1``)."""

    method: str
    """Canonical method name (``"ahc"`` / ``"hdbscan"``)."""

    backend: str
    """Concrete library that ran the fit (``"cuml"`` / ``"sklearn"`` /
    ``"hdbscan_cpu"``). Drives the dashboard backend chip."""

    distances: np.ndarray | None = None
    """Optional float32 per-crop cosine distance to the assigned centroid,
    aligned 1:1 with ``labels``. IVF populates this (drives the labeler's
    outlier sort / review queue); label-only methods (AHC, HDBSCAN) leave
    it ``None`` since they have no centroid."""

    params: dict[str, Any] = field(default_factory=dict)
    """Method-specific knobs echoed for reproducibility
    (distance_threshold, min_cluster_size, …)."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Method-specific telemetry (probabilities histogram, persistence,
    fallback reason …). Free-form; the dashboard tolerates unknown keys."""


@runtime_checkable
class ClusterMethod(Protocol):
    """Protocol every clustering backend implements."""

    name: ClassVar[str]

    async def fit_predict(
        self,
        embeddings: np.ndarray,
        *,
        backend_info: BackendInfo,
        progress: Any = None,
    ) -> ClusterResult: ...


__all__ = ['ClusterMethod', 'ClusterResult']
