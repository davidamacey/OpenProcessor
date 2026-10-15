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
    faiss: Any
    _indexes: dict[ClusterIndex, FaissIvfIndex]
    _training_metadata: dict[ClusterIndex, dict]

    def _save_index(self, index_name: ClusterIndex) -> None:
        """Save index to disk."""
        index = self._indexes.get(index_name)
        if index is None:
            return

        index_path = self.index_dir / f'{index_name.value}.index'
        self.faiss.write_index(index, str(index_path))

        # Save metadata
        metadata_path = self.index_dir / f'{index_name.value}.meta.npy'
        np.save(
            metadata_path, np.asarray(self._training_metadata.get(index_name, {}), dtype=object)
        )

        logger.info(f'Saved {index_name} index to {index_path}')

    def load_index(self, index_name: ClusterIndex) -> bool:
        """
        Load index from disk.

        Returns:
            True if loaded successfully, False if not found
        """
        index_path = self.index_dir / f'{index_name.value}.index'
        metadata_path = self.index_dir / f'{index_name.value}.meta.npy'

        if not index_path.exists():
            logger.warning(f'No saved index found for {index_name}')
            return False

        try:
            index = self.faiss.read_index(str(index_path))
            self._indexes[index_name] = index

            if metadata_path.exists():
                self._training_metadata[index_name] = np.load(
                    metadata_path, allow_pickle=True
                ).item()

            logger.info(f'Loaded {index_name} index from {index_path}')
            return True
        except Exception as e:
            logger.error(f'Failed to load {index_name} index: {e}')
            return False
