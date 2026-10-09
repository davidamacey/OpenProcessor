"""Shared singletons and index settings builders for the curation OpenSearch client."""

from __future__ import annotations

from typing import Any

from src.config import get_curation_config, get_region_fields
from src.core.logging import get_logger


logger = get_logger(__name__)


# =============================================================================
# Module-level config singletons
# =============================================================================

config = get_curation_config()
F = get_region_fields()


# =============================================================================
# Index schema builders
# =============================================================================


def _knn_field(dim: int = config.embedding_dim) -> dict[str, Any]:
    """Standard cosine k-NN HNSW field (FAISS engine)."""
    return {
        'type': 'knn_vector',
        'dimension': dim,
        'method': {
            'name': 'hnsw',
            'space_type': 'cosinesimil',
            'engine': 'faiss',
            'parameters': {
                'ef_construction': config.hnsw_ef_construction,
                'm': config.hnsw_m,
            },
        },
    }


def _knn_settings() -> dict[str, Any]:
    return {
        'index': {
            'number_of_shards': 1,
            'number_of_replicas': 0,
            'knn': True,
            # Derived source stays ON (the OpenSearch 3.x default): vectors are
            # not duplicated as JSON text in `_source` (about 3x smaller). Its
            # one trap: a search whose `_source` includes the bare path of the
            # nested per-box vector field returns the number 1 for each vector;
            # use region_box_embeddings.box_vector_source_includes instead.
        },
    }


def _plain_settings() -> dict[str, Any]:
    return {
        'index': {
            'number_of_shards': 1,
            'number_of_replicas': 0,
        },
    }


def _is_recoverable_mapping_conflict(msg: str) -> bool:
    """Classify a ``put_mapping`` failure as permanent-and-harmless vs. real.

    The ``ensure_items_*`` helpers below run on every cold start to
    additively backfill fields onto long-lived deployments. Two error
    shapes are BOTH expected, permanent conditions once a live index has
    drifted from the current schema code — not a fresh incident each
    restart, and never fatal to boot:

    * ``mapper_parsing_exception`` / "already exists" — a genuine field
      name collision (should not happen in production, but is harmless).
    * ``illegal_argument_exception`` — OpenSearch's error type for two
      immutable-setting conflicts: a field whose type already differs from
      what an ``ensure_*`` helper wants (a type change requires a full
      reindex), and setting kNN method params on an index whose
      ``index.knn`` is still ``false`` (pre-kNN-migration indices). Both
      require a reindex to actually resolve, not a retry — logging them at
      ``error`` on every single restart would be misleading operational
      noise, not an indication anything was newly broken.
    """
    return (
        'mapper_parsing_exception' in msg
        or 'already exists' in msg
        or 'illegal_argument_exception' in msg
    )
