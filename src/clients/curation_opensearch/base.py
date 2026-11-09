"""Shared singletons and index settings builders for the curation OpenSearch client."""

from __future__ import annotations

from typing import Any

from src.config import get_curation_config, get_region_fields
from src.core.logging import get_logger
from src.domain.embedding_spaces import KnnMethodSpec


logger = get_logger(__name__)


# =============================================================================
# Module-level config singletons
# =============================================================================

config = get_curation_config()
F = get_region_fields()


# =============================================================================
# Index schema builders
# =============================================================================


def default_knn_spec() -> KnnMethodSpec:
    """The k-NN method of a new project field: Lucene HNSW, cosine, sized from the config.

    Lucene stores the graph with the segment (no native graph memory), takes half the disk of
    faiss for the same vectors and keeps recall near 1.0 under the filters the review and
    search routes send (faiss with a filter fell to 0.91 after a merge). Its cosine ``_score``
    equals faiss's, so ``min_score`` is unchanged. Indexes created before this default keep
    their faiss fields: the engine of a field is fixed at creation.
    """
    return KnnMethodSpec(m=config.hnsw_m, ef_construction=config.hnsw_ef_construction)


def knn_field_body(spec: KnnMethodSpec, dim: int) -> dict[str, Any]:
    """The ``knn_vector`` mapping of one field from its method spec and dimension.

    OpenSearch k-NN vector docs: https://docs.opensearch.org/latest/field-types/supported-field-types/knn-vector/
    Optional spec members (``mode``, ``compression_level``, ``data_type``) appear only when set,
    so a spec with none of them renders the exact body the fixed builder rendered before.
    """
    body: dict[str, Any] = {
        'type': 'knn_vector',
        'dimension': dim,
        'method': {
            'name': 'hnsw',
            'space_type': spec.space_type,
            'engine': spec.engine,
            'parameters': {
                'ef_construction': spec.ef_construction,
                'm': spec.m,
            },
        },
    }
    if spec.data_type is not None:
        body['data_type'] = spec.data_type
    if spec.mode is not None:
        body['mode'] = spec.mode
    if spec.compression_level is not None:
        body['compression_level'] = spec.compression_level
    return body


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
