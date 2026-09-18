"""cuML GPU HDBSCAN for the residual pool, CPU ``hdbscan`` fallback.

HDBSCAN became the default residual-pool clusterer on 2026-05-22 after
the sklearn AHC path stalled the worker heartbeat on n=347k (the C-level
merge loop holds the GIL long enough that the heartbeat asyncio task
never gets scheduled).

Why HDBSCAN here:

* No ``distance_threshold`` or ``n_clusters`` knob — picks the most
  stable cut from the full hierarchy automatically. The only thing
  the operator sets is ``min_cluster_size`` ("smallest group I'm
  willing to call a cluster").
* GPU-native (cuML) — wall time at n=347k is single-digit minutes vs
  the multi-hour estimate for sklearn AHC.
* Emits ``-1`` (noise) for genuinely-ambiguous crops — the existing
  API already maps ``cluster_id < 0`` to ``cluster_kind=unassigned``,
  so noise points surface in the labeler as "needs human" rather than
  getting forced into a wrong cluster.

Metric note: cuML's GPU HDBSCAN supports ``metric='euclidean'`` only.
PE embeddings are L2-normalized, so for any two vectors ``a, b``:

    ||a - b||^2 = 2 - 2 * cos(a, b)

i.e. euclidean distance is a monotone function of cosine distance on
the unit sphere. HDBSCAN is rank-based (stability of the cluster
hierarchy depends on the *order* of distances, not their absolute
scale), so the chosen cut is identical to what a cosine-metric run
would produce. The CPU ``hdbscan`` package accepts cosine directly;
we still pass euclidean to keep results bit-comparable between paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from src.core.logging import get_logger
from src.services.curation.clustering.methods.base import ClusterResult


if TYPE_CHECKING:
    from src.services.curation.clustering.backend import BackendInfo


logger = get_logger(__name__)


HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_CLUSTER_SELECTION_METHOD = 'eom'


class HDBSCANMethod:
    """HDBSCAN with cuML GPU dispatch + CPU fallback."""

    name: ClassVar[str] = 'hdbscan'

    def __init__(
        self,
        *,
        min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples: int | None = None,
        cluster_selection_method: str = HDBSCAN_CLUSTER_SELECTION_METHOD,
    ) -> None:
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples) if min_samples is not None else self.min_cluster_size
        self.cluster_selection_method = cluster_selection_method

    async def fit_predict(
        self,
        embeddings: np.ndarray,
        *,
        backend_info: BackendInfo,
        progress: Any = None,
    ) -> ClusterResult:
        import asyncio as _asyncio

        if progress is not None:
            progress.raise_if_cancelled()

        if backend_info.name == 'gpu':
            try:
                labels, extra = await _asyncio.to_thread(self._fit_gpu, embeddings)
                logger.info(
                    'legacy_hdbscan_gpu_done',
                    n=len(embeddings),
                    n_clusters=extra.get('n_clusters'),
                    n_noise=extra.get('n_noise'),
                )
                return ClusterResult(
                    labels=labels,
                    method=self.name,
                    backend='cuml',
                    params=self._params(),
                    extra=extra,
                )
            except Exception as exc:
                logger.warning('legacy_hdbscan_gpu_failed_fallback_cpu', error=str(exc))

        labels, extra = await _asyncio.to_thread(self._fit_cpu, embeddings)
        logger.info(
            'legacy_hdbscan_cpu_done',
            n=len(embeddings),
            n_clusters=extra.get('n_clusters'),
            n_noise=extra.get('n_noise'),
        )
        return ClusterResult(
            labels=labels,
            method=self.name,
            backend='hdbscan_cpu',
            params=self._params(),
            extra=extra,
        )

    def _params(self) -> dict[str, Any]:
        return {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'cluster_selection_method': self.cluster_selection_method,
        }

    def _fit_gpu(self, embeddings: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        # cuML HDBSCAN is GPU-resident end-to-end (single-linkage tree,
        # condensation, stability selection). metric='euclidean' is the
        # only supported option; equivalent to cosine on L2-normalized
        # input (see module docstring).
        from cuml.cluster import (
            HDBSCAN as cuHDBSCAN,  # type: ignore[import-not-found]  # noqa: N811
        )

        # Ensure contiguous float32; cuML copies anyway but skipping the
        # internal conversion avoids a transient double-buffer in VRAM.
        x = np.ascontiguousarray(embeddings, dtype=np.float32)
        model = cuHDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
            metric='euclidean',
        )
        raw = model.fit_predict(x)
        labels = np.asarray(
            raw.get() if hasattr(raw, 'get') else raw,
            dtype=np.int64,
        )
        extra = _summarize_labels(labels)
        # Pull probabilities when cuML exposes them — useful for
        # downstream confidence-aware sorting in the labeler.
        probs_attr = getattr(model, 'probabilities_', None)
        if probs_attr is not None:
            try:
                probs = np.asarray(
                    probs_attr.get() if hasattr(probs_attr, 'get') else probs_attr,
                    dtype=np.float32,
                )
                extra['probabilities_mean'] = float(probs.mean())
                extra['probabilities_min'] = float(probs.min())
            except Exception as exc:
                logger.debug('legacy_hdbscan_probs_extract_failed', error=str(exc))
        return labels, extra

    def _fit_cpu(self, embeddings: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            import hdbscan  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "CPU HDBSCAN fallback requires the 'hdbscan' package; "
                'install it or run on a host with cuML available'
            ) from exc

        model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
            metric='euclidean',
            core_dist_n_jobs=-1,
        )
        labels = model.fit_predict(embeddings).astype(np.int64)
        extra = _summarize_labels(labels)
        if getattr(model, 'probabilities_', None) is not None:
            probs = np.asarray(model.probabilities_, dtype=np.float32)
            extra['probabilities_mean'] = float(probs.mean())
            extra['probabilities_min'] = float(probs.min())
        return labels, extra


def _summarize_labels(labels: np.ndarray) -> dict[str, Any]:
    """Compact stats for the auto-label job summary / dashboard."""
    unique, counts = np.unique(labels, return_counts=True)
    n_noise = int(counts[unique == -1].sum()) if (-1 in unique) else 0
    real_counts = counts[unique != -1]
    n_clusters = len(real_counts)
    largest = int(real_counts.max()) if real_counts.size else 0
    n_assigned = int(len(labels) - n_noise)
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_pct': (n_noise / len(labels)) if len(labels) else 0.0,
        'largest_cluster_size': largest,
        'largest_cluster_pct': (largest / n_assigned) if n_assigned else 0.0,
    }


__all__ = [
    'HDBSCAN_CLUSTER_SELECTION_METHOD',
    'HDBSCAN_MIN_CLUSTER_SIZE',
    'HDBSCANMethod',
]
