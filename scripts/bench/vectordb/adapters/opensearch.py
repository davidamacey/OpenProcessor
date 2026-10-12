"""OpenSearch k-NN adapter: the shipped index shape and every tuning arm of the plan.

The field layout mirrors ``src/clients/curation_opensearch/base.py`` (``knn_vector`` with the
faiss HNSW method, cosine, one shard, no replica); only the scalar fields the benchmark filters
on are mapped. The query is the product's: a ``knn`` clause with the filter inside it.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from scripts.bench.vectordb.adapters.base import Adapter, Meta, run_threads
from scripts.bench.vectordb.adapters.os_rest import Rest


if TYPE_CHECKING:
    import numpy as np

    from scripts.bench.vectordb.vdb_lib import FilterSpec


INDEX = 'vdb_bench'
FIELD_TYPES = {
    'project': 'integer',
    'class_name': 'keyword',
    'class_validated': 'boolean',
    'test_holdout': 'boolean',
    'review_dismissed': 'boolean',
    'cluster_id': 'integer',
    'sel': 'integer',
}


def filter_clauses(flt: FilterSpec) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = [{'term': {f: v}} for f, v in flt.eq]
    clauses += [{'term': {f: False}} for f in flt.is_false]
    if flt.sel_lt is not None:
        clauses.append({'range': {'sel': {'lt': flt.sel_lt}}})
    return clauses


class OpenSearchAdapter(Adapter):
    engine = 'opensearch'
    container = 'opensearch'
    data_path = '/usr/share/opensearch/data'

    def __init__(self, arm: dict[str, Any], endpoint: str) -> None:
        super().__init__(arm, endpoint)
        self.client = Rest(endpoint)

    def method(self) -> dict[str, Any]:
        """The ``method`` block of the vector field for this arm."""
        engine = self.arm.get('knn_engine', 'faiss')
        params: dict[str, Any] = {
            'm': self.arm.get('m', 16),
            'ef_construction': self.arm.get('ef_construction', 512),
        }
        if self.arm.get('encoder') == 'fp16':
            params['encoder'] = {'name': 'sq', 'parameters': {'bits': 16}}
        return {
            'name': 'hnsw',
            'space_type': 'cosinesimil',
            'engine': engine,
            'parameters': params,
        }

    def vector_field(self, dim: int) -> dict[str, Any]:
        field: dict[str, Any] = {'type': 'knn_vector', 'dimension': dim}
        if self.arm.get('mode'):
            field['mode'] = self.arm['mode']
            field['compression_level'] = self.arm['compression_level']
            if self.arm.get('ef_construction') or self.arm.get('m'):
                field['method'] = {
                    'name': 'hnsw',
                    'engine': 'faiss',
                    'parameters': {
                        'm': self.arm.get('m', 16),
                        'ef_construction': self.arm.get('ef_construction', 512),
                    },
                }
        else:
            field['method'] = self.method()
        return field

    def create(self, dim: int) -> None:
        if self.client.exists(INDEX):
            self.client.delete(INDEX)
        settings: dict[str, Any] = {
            'number_of_shards': self.arm.get('shards', 1),
            'number_of_replicas': 0,
            'knn': True,
        }
        if 'refresh_interval' in self.arm:
            settings['refresh_interval'] = self.arm['refresh_interval']
        props = {f: {'type': t} for f, t in FIELD_TYPES.items()}
        props['vec'] = self.vector_field(dim)
        self.client.create(
            INDEX, {'settings': {'index': settings}, 'mappings': {'properties': props}}
        )

    def encode(self, vectors: np.ndarray, meta: Meta, ids: np.ndarray) -> str:
        lines: list[str] = []
        for pos, row_id in enumerate(ids):
            doc: dict[str, Any] = {'vec': vectors[pos].tolist()}
            for field in FIELD_TYPES:
                value = meta[field][pos]
                doc[field] = value.item() if hasattr(value, 'item') else value
            lines.append(json.dumps({'index': {'_index': INDEX, '_id': str(int(row_id))}}))
            lines.append(json.dumps(doc))
        return '\n'.join(lines) + '\n'

    def send(self, payload: Any) -> None:
        self.send_bulk(payload)

    def send_bulk(self, body: Any) -> None:
        for attempt in range(8):
            resp = self.client.bulk(body=body)
            if not resp.get('errors'):
                return
            retry = [it for it in resp['items'] if next(iter(it.values())).get('status') == 429]
            if not retry or attempt == 7:
                raise RuntimeError(f'bulk failed: {resp["items"][0]}')
            time.sleep(1 + attempt)

    def finish_load(self) -> None:
        self.client.refresh(INDEX)
        if self.arm.get('refresh_interval') == '-1':
            self.client.put_settings(INDEX, {'index': {'refresh_interval': '1s'}})
        if self.arm.get('merge', True):
            self.client.forcemerge(INDEX)
        self.client.refresh(INDEX)
        self.client.warmup(INDEX)

    def search(
        self, vector: np.ndarray, k: int, flt: FilterSpec, params: dict[str, Any]
    ) -> list[int]:
        knn: dict[str, Any] = {'vector': vector.tolist(), 'k': k}
        clauses = filter_clauses(flt)
        if clauses:
            knn['filter'] = {'bool': {'filter': clauses}}
        method_params = {}
        if 'ef_search' in params:
            method_params['ef_search'] = (
                max(params['ef_search'], k)
                if self.arm.get('knn_engine') == 'lucene'
                else params['ef_search']
            )
        if method_params:
            knn['method_parameters'] = method_params
        if 'oversample' in params:
            knn['rescore'] = {'oversample_factor': params['oversample']}
        body = {'size': k, 'query': {'knn': {'vec': knn}}, '_source': False}
        resp = self.client.search(INDEX, body)
        return [int(h['_id']) for h in resp['hits']['hits']]

    def update_field(self, ids: np.ndarray, field: str, values: np.ndarray, threads: int) -> None:
        def work(a: int, b: int) -> None:
            body: list[dict[str, Any]] = []
            for pos in range(a, b):
                body.append({'update': {'_index': INDEX, '_id': str(int(ids[pos]))}})
                body.append({'doc': {field: int(values[pos])}})
            self.send_bulk(body)

        run_threads(work, len(ids), 1000, threads)
        self.client.refresh(INDEX)

    def delete_where(self, flt: FilterSpec) -> None:
        self.client.delete_by_query(INDEX, {'query': {'bool': {'filter': filter_clauses(flt)}}})

    def settle(self) -> None:
        self.client.forcemerge(INDEX)
        self.client.flush(INDEX)

    def bytes_on_disk(self) -> int:
        stats = self.client.stats(INDEX, 'store')
        return int(stats['_all']['primaries']['store']['size_in_bytes'])

    def engine_stats(self) -> dict[str, Any]:
        """Native graph memory, JVM heap, segment layout and merge/refresh time of the index."""
        knn = self.client.knn_stats()
        node = next(iter(knn['nodes'].values()))
        jvm = self.client.jvm()
        heap = next(iter(jvm.values()))['jvm']['mem']
        segs = self.client.segments(INDEX)
        stats = self.client.stats(INDEX, 'merge,refresh,segments')
        prim = stats['_all']['primaries']
        return {
            'knn_graph_memory_kb': node.get('graph_memory_usage'),
            'knn_graph_memory_pct': node.get('graph_memory_usage_percentage'),
            'heap_used_bytes': heap['heap_used_in_bytes'],
            'heap_max_bytes': heap['heap_max_in_bytes'],
            'segments': len(segs),
            'segment_bytes': sorted((int(x['size']) for x in segs), reverse=True)[:8],
            'merge_total_ms': prim['merges']['total_time_in_millis'],
            'merge_total_docs': prim['merges']['total_docs'],
            'refresh_total_ms': prim['refresh']['total_time_in_millis'],
            'refresh_count': prim['refresh']['total'],
        }

    def count(self) -> int:
        return self.client.count(INDEX)

    def drop(self) -> None:
        if self.client.exists(INDEX):
            self.client.delete(INDEX)

    def close(self) -> None:
        self.client.close()
