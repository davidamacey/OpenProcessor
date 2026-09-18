"""Persistent FAISS IVF centroid store for the residual pool.

Holds the k-means centroids that partition the residual (unconfident)
embedding pool into candidate buckets. Persisted to a shared volume so
two very different callers can use the same centroids:

* the **auto-label worker** trains them (periodic recluster) and writes
  them here;
* the **ingest path** (yolo-api) loads them once and assigns every new
  residual crop to its nearest centroid at ingest time — no batch wait.

Both processes mount the same configured state directory
(``CurationConfig.state_dir``, default ``/var/lib/openprocessor``), so the
store lives at ``<state_dir>/ivf_residuals/``:

    centroids.faiss   IndexFlatL2 over the K centroid vectors
    metadata.json     {trained_at, n_clusters, embedding_dim,
                       sample_size, n_trained_on, kmeans_obj, backend}

Assignment is a nearest-centroid search (IndexFlatL2). The returned
label is the centroid index ``0..K-1``; callers add
``RESIDUAL_CLUSTER_ID_OFFSET`` to map it into the candidate id space.

The store is deliberately dependency-light: it imports faiss lazily so
modules that only need the *paths* (e.g. for a readiness probe) don't
pull faiss in.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_curation_config
from src.core.logging import get_logger


logger = get_logger(__name__)


IVF_STORE_DIR = Path(get_curation_config().state_dir) / 'ivf_residuals'
CENTROIDS_PATH = IVF_STORE_DIR / 'centroids.faiss'
METADATA_PATH = IVF_STORE_DIR / 'metadata.json'
# Primary-subject clustering gate, persisted decoupled from the centroids so
# both the worker (full recluster) and ingest (per-crop assign) apply the
# SAME policy. Empty / absent file = no gate (cluster the full residual pool).
GATE_PATH = IVF_STORE_DIR / 'gate.json'


class IVFCentroidStore:
    """Load / save / assign against persisted FAISS IVF centroids."""

    def __init__(self, store_dir: Path | str | None = None) -> None:
        self._dir = Path(store_dir) if store_dir is not None else IVF_STORE_DIR
        self._centroids_path = self._dir / 'centroids.faiss'
        self._metadata_path = self._dir / 'metadata.json'
        self._gate_path = self._dir / 'gate.json'
        self._index: Any = None  # lazily-loaded faiss.IndexFlatL2
        self._metadata: dict[str, Any] = {}

    # -- existence / metadata ---------------------------------------------

    def is_trained(self) -> bool:
        """True if a persisted centroid index exists on disk."""
        return self._centroids_path.is_file()

    @property
    def metadata(self) -> dict[str, Any]:
        if not self._metadata and self._metadata_path.is_file():
            try:
                self._metadata = json.loads(self._metadata_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning('kb_ivf_store_metadata_read_failed', error=str(exc))
        return self._metadata

    @property
    def n_clusters(self) -> int:
        return int(self.metadata.get('n_clusters', 0))

    @property
    def embedding_dim(self) -> int:
        return int(self.metadata.get('embedding_dim', 0))

    # -- load / save ------------------------------------------------------

    def load(self) -> bool:
        """Load the centroid index into memory. Returns success."""
        if self._index is not None:
            return True
        if not self._centroids_path.is_file():
            return False
        try:
            import faiss

            self._index = faiss.read_index(str(self._centroids_path))
            logger.info(
                'kb_ivf_store_loaded',
                n_clusters=self._index.ntotal,
                dim=self._index.d,
                path=str(self._centroids_path),
            )
            return True
        except Exception as exc:
            logger.warning('kb_ivf_store_load_failed', error=str(exc))
            self._index = None
            return False

    def save(self, centroids: np.ndarray, *, metadata: dict[str, Any]) -> None:
        """Persist centroids (K x dim) + metadata atomically."""
        import faiss

        centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        k, dim = centroids.shape
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)

        self._dir.mkdir(parents=True, exist_ok=True)
        # Atomic writes: write to a temp file in the same dir, then rename.
        # A half-written centroids.faiss must never be loadable.
        meta = {
            **metadata,
            'n_clusters': int(k),
            'embedding_dim': int(dim),
            'trained_at': datetime.now(UTC).isoformat(),
        }
        with tempfile.NamedTemporaryFile(dir=self._dir, suffix='.faiss.tmp', delete=False) as tf:
            tmp_index = tf.name
        faiss.write_index(index, tmp_index)
        Path(tmp_index).replace(self._centroids_path)

        with tempfile.NamedTemporaryFile(
            'w', dir=self._dir, suffix='.json.tmp', delete=False
        ) as tf:
            json.dump(meta, tf, indent=2, default=str)
            tmp_meta = tf.name
        Path(tmp_meta).replace(self._metadata_path)

        self._index = index
        self._metadata = meta
        logger.info('kb_ivf_store_saved', n_clusters=k, dim=dim, path=str(self._centroids_path))

    def update_metadata(self, **fields: Any) -> None:
        """Merge ``fields`` into the persisted metadata without touching centroids.

        Used to record post-hoc facts about the last training run (e.g.
        ``trained_mode``) that ``save()`` itself doesn't know — callers
        outside the k-means fit decide those. No-op if no metadata exists
        yet (nothing trained).
        """
        if not self._metadata_path.is_file():
            return
        meta = dict(self.metadata)
        meta.update(fields)
        self._dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            'w', dir=self._dir, suffix='.json.tmp', delete=False
        ) as tf:
            json.dump(meta, tf, indent=2, default=str)
            tmp_meta = tf.name
        Path(tmp_meta).replace(self._metadata_path)
        self._metadata = meta

    # -- train ------------------------------------------------------------

    def train(
        self,
        sample: np.ndarray,
        *,
        n_clusters: int,
        niter: int,
        nredo: int,
        use_gpu: bool,
    ) -> tuple[np.ndarray, float | None]:
        """Run FAISS k-means on ``sample``; return ``(centroids, final_obj)``.

        Does NOT persist — caller decides via :meth:`save`. ``sample`` is
        the training subset; for the full-pool assign use
        :meth:`assign_batch` after saving.
        """
        import faiss

        x = np.ascontiguousarray(sample, dtype=np.float32)
        dim = x.shape[1]
        gpu = use_gpu and faiss.get_num_gpus() > 0
        # spherical=True: re-normalize centroids to unit length each
        # iteration so k-means optimizes COSINE, matching the PE embedding
        # training objective. With unit centroids and unit queries the
        # IndexFlatL2 assignment below ranks identically to cosine
        # (||x-c||^2 = 2 - 2*cos, constant offset), so no assignment-side
        # change is needed. Takes effect on the next full retrain.
        kmeans = faiss.Kmeans(
            dim,
            n_clusters,
            niter=niter,
            nredo=nredo,
            verbose=False,
            gpu=gpu,
            seed=42,
            spherical=True,
        )
        kmeans.train(x)
        # kmeans.index is a flat index holding the K centroids; reconstruct
        # them as an (n_clusters, dim) array. Version-robust (works across
        # FAISS releases where kmeans.centroids' Python type varies).
        centroids = kmeans.index.reconstruct_n(0, n_clusters)
        centroids = np.asarray(centroids, dtype=np.float32).reshape(n_clusters, dim)
        # Safety: guarantee unit-norm centroids regardless of FAISS version
        # behavior, so the cosine-equivalence of the L2 assignment holds.
        cnorms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.where(cnorms == 0, 1.0, cnorms)
        final_obj: float | None = None
        with contextlib.suppress(Exception):
            final_obj = float(kmeans.obj[-1]) if len(kmeans.obj) else None
        return np.ascontiguousarray(centroids, dtype=np.float32), final_obj

    # -- clustering gate policy -------------------------------------------

    def save_gate(self, *, max_rank: int | None, min_blur_ratio: float | None) -> None:
        """Persist the primary-subject clustering gate (or clear it).

        Both ``None`` clears the gate (full-pool clustering). Ingest +
        ``assign_only_residuals`` read this so they apply the exact policy
        the last full recluster trained under.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        if max_rank is None and min_blur_ratio is None:
            with contextlib.suppress(FileNotFoundError):
                self._gate_path.unlink()
            return
        payload = {
            'max_rank': max_rank,
            'min_blur_ratio': min_blur_ratio,
            'saved_at': datetime.now(UTC).isoformat(),
        }
        with tempfile.NamedTemporaryFile(
            'w', dir=self._dir, suffix='.json.tmp', delete=False
        ) as tf:
            json.dump(payload, tf, indent=2, default=str)
            tmp = tf.name
        Path(tmp).replace(self._gate_path)

    def load_gate(self) -> dict[str, Any]:
        """Return the persisted gate ``{max_rank, min_blur_ratio}`` or ``{}``."""
        if not self._gate_path.is_file():
            return {}
        try:
            return json.loads(self._gate_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('kb_ivf_store_gate_read_failed', error=str(exc))
            return {}

    # -- assign -----------------------------------------------------------

    def assign_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign each row to its nearest centroid. Returns int64 labels."""
        labels, _ = self.assign_batch_with_distances(embeddings)
        return labels

    def assign_batch_with_distances(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Assign rows to nearest centroid; return ``(labels, cosine_dist)``.

        The IndexFlatL2 search returns *squared* L2 distance. Centroids are
        unit-norm (spherical k-means) and callers pass unit-norm queries, so
        ``||x-c||^2 = 2 - 2*cos(x,c)`` → ``cosine_dist = ||x-c||^2 / 2`` in
        ``[0, 2]``. This matches the ``cluster_distance`` semantics the review
        queue already sorts on (outlier threshold ~0.35). Pre-spherical
        centroids make this approximate, but it's only used for outlier
        ranking and self-corrects on the first spherical retrain.
        """
        if self._index is None and not self.load():
            raise RuntimeError('IVF centroid store not trained / not loadable')
        x = np.ascontiguousarray(embeddings, dtype=np.float32)
        sq_l2, assignments = self._index.search(x, 1)
        labels = assignments.reshape(-1).astype(np.int64)
        cosine_dist = np.clip(sq_l2.reshape(-1).astype(np.float32) / 2.0, 0.0, 2.0)
        return labels, cosine_dist

    def assign_one(self, embedding: np.ndarray) -> int:
        """Assign a single embedding to its nearest centroid index."""
        return self.assign_one_with_distance(embedding)[0]

    def assign_one_with_distance(self, embedding: np.ndarray) -> tuple[int, float]:
        """Assign one embedding; return ``(centroid_index, cosine_dist)``."""
        vec = np.ascontiguousarray(embedding, dtype=np.float32).reshape(1, -1)
        labels, dists = self.assign_batch_with_distances(vec)
        return int(labels[0]), float(dists[0])


__all__ = [
    'CENTROIDS_PATH',
    'GATE_PATH',
    'IVF_STORE_DIR',
    'METADATA_PATH',
    'IVFCentroidStore',
]
