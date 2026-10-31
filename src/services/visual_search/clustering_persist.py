"""Disk persistence of the FAISS cluster indexes.

:class:`IndexPersistenceMixin` is mixed into
:class:`~src.services.visual_search.clustering.ClusteringService`, which provides the
index cache, the training metadata, ``index_dir`` and the lazily imported ``faiss``.
"""

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.config.cluster_index import ClusterIndex


logger = logging.getLogger(__name__)


class FaissIvfIndex(Protocol):
    """The slice of a FAISS ``IndexIVFFlat`` this package touches (faiss ships no stubs)."""

    ntotal: int
    nlist: int
    is_trained: bool
    verbose: bool
    quantizer: Any
    invlists: Any

    def train(self, x: np.ndarray) -> None: ...

    def add(self, x: np.ndarray) -> None: ...


class IndexPersistenceMixin:
    """Save and load one index and its training metadata under ``index_dir``."""

    index_dir: Path
    embedding_dim: int
    faiss: Any
    _unloadable: set[ClusterIndex]
    _indexes: dict[ClusterIndex, FaissIvfIndex]
    _training_metadata: dict[ClusterIndex, dict]

    def _save_index(self, index_name: ClusterIndex) -> None:
        """Save index and metadata; each file is written to a temp name, then renamed."""
        index = self._indexes.get(index_name)
        if index is None:
            return

        index_path = self.index_dir / f'{index_name.value}.index'
        metadata_path = self.index_dir / f'{index_name.value}.meta.npy'
        tmp_index = index_path.with_name(index_path.name + '.tmp')
        tmp_meta = metadata_path.with_name(f'{index_name.value}.meta.tmp.npy')

        np.save(tmp_meta, np.asarray(self._training_metadata.get(index_name, {}), dtype=object))
        self.faiss.write_index(index, str(tmp_index))
        tmp_meta.replace(metadata_path)
        tmp_index.replace(index_path)

        self._unloadable.discard(index_name)
        logger.info(f'Saved {index_name} index to {index_path}')

    def has_saved_index(self, index_name: ClusterIndex) -> bool:
        """True when a persisted index file exists and has not already failed to load."""
        return (
            index_name not in self._unloadable
            and (self.index_dir / f'{index_name.value}.index').exists()
        )

    def load_index(self, index_name: ClusterIndex) -> bool:
        """
        Load index from disk.

        Fails closed: a corrupt, untrained or dimension-mismatched file is logged, kept out of
        the cache and remembered as unloadable, so callers see an untrained index.

        Returns:
            True if loaded successfully, False if not found or unusable
        """
        index_path = self.index_dir / f'{index_name.value}.index'
        metadata_path = self.index_dir / f'{index_name.value}.meta.npy'

        if not index_path.exists():
            logger.warning(f'No saved index found for {index_name}')
            return False

        try:
            index = self.faiss.read_index(str(index_path))
            if getattr(index, 'd', None) != self.embedding_dim or not index.is_trained:
                raise ValueError(
                    f'index d={getattr(index, "d", None)} trained={index.is_trained}, '
                    f'expected d={self.embedding_dim} trained'
                )
            metadata: dict = {}
            if metadata_path.exists():
                metadata = np.load(metadata_path, allow_pickle=True).item()
        except Exception as e:
            logger.error(f'Failed to load {index_name} index from {index_path}: {e}')
            self._unloadable.add(index_name)
            return False

        self._indexes[index_name] = index
        self._training_metadata[index_name] = metadata
        logger.info(f'Loaded {index_name} index from {index_path}')
        return True
