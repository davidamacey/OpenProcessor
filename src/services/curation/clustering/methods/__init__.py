"""Registry of clustering backends for the residual pool.

Pick a method by name; the registry returns an instance ready to ``await
.fit_predict()``. Default history:

* 2026-05-22 — flipped from AHC to HDBSCAN after sklearn AHC starved
  the worker heartbeat at n=347k.
* 2026-05-23 — flipped from HDBSCAN to IVF after a param sweep proved
  HDBSCAN's density-based selection collapses on continuously-dense
  vehicle embeddings (every combo: either one mega-cluster or 100 %
  noise). IVF (k-means partitioning) gives 512 balanced buckets a
  human can browse without tuning.

HDBSCAN + AHC remain available as ``clustering_method="hdbscan"`` /
``"ahc"`` for the small-n refine endpoint or future use cases where
the data has actual density structure.
"""

from __future__ import annotations

from src.services.curation.clustering.methods.ahc import AHCMethod
from src.services.curation.clustering.methods.base import ClusterMethod, ClusterResult
from src.services.curation.clustering.methods.hdbscan import HDBSCANMethod
from src.services.curation.clustering.methods.ivf import IVFMethod


DEFAULT_METHOD = 'ivf'

_METHODS: dict[str, type[ClusterMethod]] = {
    AHCMethod.name: AHCMethod,
    HDBSCANMethod.name: HDBSCANMethod,
    IVFMethod.name: IVFMethod,
}


def get_method(name: str | None = None, **kwargs) -> ClusterMethod:
    """Resolve a clustering method by name; default → IVF (see DEFAULT_METHOD)."""
    key = (name or DEFAULT_METHOD).lower()
    if key not in _METHODS:
        raise ValueError(f'unknown clustering method {name!r}; valid: {sorted(_METHODS)}')
    return _METHODS[key](**kwargs)


def available_methods() -> list[str]:
    return sorted(_METHODS)


__all__ = [
    'DEFAULT_METHOD',
    'AHCMethod',
    'ClusterMethod',
    'ClusterResult',
    'HDBSCANMethod',
    'IVFMethod',
    'available_methods',
    'get_method',
]
