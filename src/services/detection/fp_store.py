"""Persisted region false-positive centroids.

A small FAISS ``IndexFlatL2`` over per-sub-type false-positive
centroids, plus a metadata sidecar. Mirrors the atomic
temp-write + rename pattern used by other centroid stores in this
codebase, so a half-written index is never loadable. Both the API
process and any background worker read/write the same
``CurationConfig.state_dir``, so the store stays consistent across
uvicorn workers.

The on-disk subdirectory name is derived from
:class:`~src.config.RegionFields.prefix` (default ``region`` ->
``region_fp``) rather than hardcoded, so a deployment with an existing
store under a different prefix (e.g. a proprietary-dataset overlay
using ``plate_fp``) can point at it without a code change — see
``docs/design/curation_design_rationale.md`` §4.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config import CurationConfig, RegionFields, get_curation_config, get_region_fields
from src.core.logging import get_logger


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger(__name__)


def fp_store_dir(
    config: CurationConfig | None = None,
    fields: RegionFields | None = None,
) -> Path:
    """Resolve the false-positive centroid store directory for a deployment.

    ``{config.state_dir}/{fields.prefix}_fp`` — e.g. the OSS default is
    ``/var/lib/openprocessor/region_fp``.
    """
    cfg = config or get_curation_config()
    rf = fields or get_region_fields()
    return cfg.state_dir / f'{rf.prefix}_fp'


class FalsePositiveCentroidStore:
    """FAISS-backed store of false-positive sub-type centroids.

    The directory defaults to :func:`fp_store_dir` resolved against the
    module-level default :class:`CurationConfig` / :class:`RegionFields`
    singletons, but a deployment-specific pair (or an explicit
    ``directory``) can be injected — this is what makes the store's
    location a config flip rather than a hardcoded path.
    """

    def __init__(
        self,
        directory: Path | None = None,
        *,
        config: CurationConfig | None = None,
        fields: RegionFields | None = None,
    ) -> None:
        self._dir = directory if directory is not None else fp_store_dir(config, fields)
        self._index: Any = None
        self.metadata: dict[str, Any] = {}

    @property
    def directory(self) -> Path:
        return self._dir

    def exists(self) -> bool:
        return (self._dir / 'centroids.faiss').is_file()

    def load(self) -> bool:
        """Load the index + metadata from disk. Returns False if not built yet."""
        import faiss

        path = self._dir / 'centroids.faiss'
        if not path.is_file():
            return False
        self._index = faiss.read_index(str(path))
        try:
            self.metadata = json.loads((self._dir / 'metadata.json').read_text())
        except Exception:
            self.metadata = {}
        return True

    def save(self, centroids: np.ndarray, metadata: dict[str, Any]) -> None:
        """Persist ``centroids`` (K x dim) + ``metadata`` atomically."""
        import faiss

        self._dir.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatL2(int(centroids.shape[1]))
        index.add(np.ascontiguousarray(centroids, dtype=np.float32))
        tmp = self._dir / 'centroids.faiss.tmp'
        faiss.write_index(index, str(tmp))
        tmp.replace(self._dir / 'centroids.faiss')
        mtmp = self._dir / 'metadata.json.tmp'
        mtmp.write_text(json.dumps(metadata))
        mtmp.replace(self._dir / 'metadata.json')
        self._index = index
        self.metadata = metadata

    def search(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(distances, subtype_indices)`` to the nearest FP centroid.

        CM-3: ``faiss.IndexFlatL2`` returns *squared* L2 distance, not L2.
        Every caller (auto-assign FP threshold, suspected-FP threshold,
        the "L2 on unit-norm" comments at the call sites) was written
        assuming plain L2, where orthogonal unit vectors are ``sqrt(2)``
        and identical vectors are 0. Left squared, a distance of 0.20
        (documented as an ~0.90 cosine cut) is actually an ~0.90 *squared*
        distance -> cosine similarity 1 - 0.20/2 = 0.90, which happened to
        read right by coincidence at small values but diverges badly
        everywhere else (e.g. the 0.35 suspected-FP threshold was really
        cosine >= 0.825, not the documented ~0.94). Taking the square root
        here makes every downstream threshold comparison correct in L2
        units without touching the call sites' math.
        """
        x = np.ascontiguousarray(embeddings, dtype=np.float32)
        dist, idx = self._index.search(x, 1)
        return np.sqrt(np.maximum(dist.reshape(-1), 0.0)), idx.reshape(-1)


__all__ = ['FalsePositiveCentroidStore', 'fp_store_dir']
