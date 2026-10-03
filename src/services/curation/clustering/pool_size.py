"""Pool-size rules shared by every clustering entry point."""

from __future__ import annotations

from typing import Any


UMAP_N_COMPONENTS = 50
UMAP_N_NEIGHBORS = 15

# Fewer residuals than this and no clusterer (UMAP, AHC, IVF) has enough
# points to be meaningful; every clustering entry point refuses below it.
MIN_RESIDUALS_FOR_CLUSTERING = 32

# Spectral init needs an eigensolver with k < N; below this margin over
# the component count it raises a TypeError inside scipy.
_UMAP_SPECTRAL_MARGIN = 10


class TooFewItemsError(Exception):
    """Fewer residual items than :data:`MIN_RESIDUALS_FOR_CLUSTERING`."""

    def __init__(self, n_items: int) -> None:
        super().__init__(
            f'{n_items} items; clustering needs at least {MIN_RESIDUALS_FOR_CLUSTERING}'
        )
        self.n_items = n_items
        self.min_items = MIN_RESIDUALS_FOR_CLUSTERING


def require_min_items(n_items: int) -> None:
    """The one clustering size guard: raise :class:`TooFewItemsError` below the floor."""
    if n_items < MIN_RESIDUALS_FOR_CLUSTERING:
        raise TooFewItemsError(n_items)


def umap_shape(n_items: int) -> dict[str, Any]:
    """Dimensionality/neighbour/init settings that are safe for ``n_items`` points.

    Components and neighbours are clamped below N; small pools use random
    init because spectral init crashes when k approaches N.
    """
    n_components = min(UMAP_N_COMPONENTS, n_items - 2)
    return {
        'n_components': n_components,
        'n_neighbors': min(UMAP_N_NEIGHBORS, n_items - 1),
        'init': 'random' if n_items < n_components + _UMAP_SPECTRAL_MARGIN else 'spectral',
    }
