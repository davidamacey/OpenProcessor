"""Arm definitions: engine, build configuration and the query-time parameter grid (cheapest first)."""

from __future__ import annotations

from typing import Any


EF = [16, 32, 64, 128, 256, 512]
# 100 is the OpenSearch default (index.knn.algo_param.ef_search), the shipped value.
OS_GRID = [{'ef_search': e} for e in (16, 32, 64, 100, 128, 256, 512)]

ARMS: dict[str, dict[str, Any]] = {
    # OpenSearch: the shipped index shape, then one change at a time.
    'os_shipped': {'store': 'opensearch', 'merge': False, 'grid': OS_GRID},
    'os_shipped_merged': {'store': 'opensearch', 'grid': OS_GRID},
    'os_efc128': {'store': 'opensearch', 'ef_construction': 128, 'grid': OS_GRID},
    'os_m32': {'store': 'opensearch', 'm': 32, 'grid': OS_GRID},
    'os_lucene': {'store': 'opensearch', 'knn_engine': 'lucene', 'grid': OS_GRID},
    'os_fp16': {'store': 'opensearch', 'encoder': 'fp16', 'grid': OS_GRID},
    'os_disk8x': {
        'store': 'opensearch',
        'mode': 'on_disk',
        'compression_level': '8x',
        'grid': OS_GRID,
    },
    'os_disk16x': {
        'store': 'opensearch',
        'mode': 'on_disk',
        'compression_level': '16x',
        'grid': OS_GRID,
    },
    'os_disk32x': {
        'store': 'opensearch',
        'mode': 'on_disk',
        'compression_level': '32x',
        'grid': OS_GRID,
    },
    'os_shards2': {'store': 'opensearch', 'shards': 2, 'grid': OS_GRID},
    # Milvus standalone.
    'milvus_hnsw': {'store': 'milvus', 'index': 'HNSW', 'grid': [{'ef': e} for e in EF]},
    'milvus_hnsw_pk': {
        'store': 'milvus',
        'index': 'HNSW',
        'partition_key': True,
        'grid': [{'ef': e} for e in EF],
    },
    'milvus_ivf_flat': {
        'store': 'milvus',
        'index': 'IVF_FLAT',
        'grid': [{'nprobe': n} for n in (16, 32, 64, 128)],
    },
    'milvus_ivf_sq8': {
        'store': 'milvus',
        'index': 'IVF_SQ8',
        'grid': [{'nprobe': n} for n in (16, 32, 64, 128)],
    },
    # Qdrant.
    'qdrant_hnsw': {'store': 'qdrant', 'grid': [{'hnsw_ef': e} for e in EF]},
    # The documented bulk path: no graph while loading, build it once afterwards.
    'qdrant_deferred': {
        'store': 'qdrant',
        'defer_index': True,
        'grid': [{'hnsw_ef': e} for e in EF],
    },
    'qdrant_int8': {
        'store': 'qdrant',
        'quantization': 'int8',
        'grid': [{'hnsw_ef': e} for e in EF],
    },
    'qdrant_int8_disk': {
        'store': 'qdrant',
        'quantization': 'int8',
        'on_disk_vectors': True,
        'grid': [{'hnsw_ef': e} for e in EF],
    },
    # pgvector.
    'pg_vector': {'store': 'pgvector', 'grid': OS_GRID},
    'pg_halfvec': {'store': 'pgvector', 'half': True, 'grid': OS_GRID},
    # LanceDB (embedded).
    'lance_ivfpq': {
        'store': 'lancedb',
        'index': 'IVF_PQ',
        'grid': [{'nprobes': n, 'refine': 10} for n in (10, 20, 50, 100)],
    },
    'lance_hnsw_sq': {
        'store': 'lancedb',
        'index': 'IVF_HNSW_SQ',
        'grid': [{'nprobes': n, 'ef': e} for n, e in ((10, 64), (20, 128), (50, 256), (100, 512))],
    },
    # Reference (not a datastore).
    'ref_flat': {'store': 'faiss_flat', 'grid': [{}]},
}
