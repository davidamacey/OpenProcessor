"""Qdrant adapter: HNSW with payload indexes, optional int8 scalar quantization, on-disk vectors."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from scripts.bench.vectordb.adapters.base import Adapter, Meta, run_threads


if TYPE_CHECKING:
    import numpy as np

    from scripts.bench.vectordb.vdb_lib import FilterSpec


COLLECTION = 'vdb_bench'
INDEXED = {
    'project': 'integer',
    'class_name': 'keyword',
    'class_validated': 'bool',
    'test_holdout': 'bool',
    'review_dismissed': 'bool',
    'cluster_id': 'integer',
    'sel': 'integer',
}


class QdrantAdapter(Adapter):
    engine = 'qdrant'
    container = 'qdrant'
    data_path = '/qdrant/storage'

    def __init__(self, arm: dict[str, Any], endpoint: str) -> None:
        super().__init__(arm, endpoint)
        from qdrant_client import QdrantClient

        host, _, port = endpoint.partition(':')
        self.client = QdrantClient(
            host=host, port=int(port), grpc_port=int(port) + 3, prefer_grpc=True, timeout=3600
        )

    def create(self, dim: int) -> None:
        from qdrant_client import models as m

        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        quant = None
        if self.arm.get('quantization') == 'int8':
            quant = m.ScalarQuantization(
                scalar=m.ScalarQuantizationConfig(type=m.ScalarType.INT8, always_ram=True)
            )
        self.client.create_collection(
            COLLECTION,
            vectors_config=m.VectorParams(
                size=dim,
                distance=m.Distance.COSINE,
                on_disk=bool(self.arm.get('on_disk_vectors', False)),
                datatype=m.Datatype.FLOAT16 if self.arm.get('fp16') else m.Datatype.FLOAT32,
            ),
            hnsw_config=m.HnswConfigDiff(
                m=0 if self.arm.get('defer_index') else self.arm.get('m', 16),
                ef_construct=self.arm.get('ef_construction', 512),
            ),
            quantization_config=quant,
        )
        schema = {
            'integer': m.PayloadSchemaType.INTEGER,
            'keyword': m.PayloadSchemaType.KEYWORD,
            'bool': m.PayloadSchemaType.BOOL,
        }
        for field, kind in INDEXED.items():
            self.client.create_payload_index(COLLECTION, field, schema[kind])

    def encode(self, vectors: np.ndarray, meta: Meta, ids: np.ndarray) -> list[Any]:
        from qdrant_client import models as m

        points = []
        for pos in range(len(vectors)):
            payload = {}
            for f in INDEXED:
                value = meta[f][pos]
                payload[f] = value.item() if hasattr(value, 'item') else value
            points.append(
                m.PointStruct(id=int(ids[pos]), vector=vectors[pos].tolist(), payload=payload)
            )
        return points

    def send(self, payload: Any) -> None:
        self.client.upsert(COLLECTION, points=payload, wait=True)

    def finish_load(self) -> None:
        if self.arm.get('defer_index'):
            from qdrant_client import models as m

            self.client.update_collection(
                COLLECTION,
                hnsw_config=m.HnswConfigDiff(
                    m=self.arm.get('m', 16), ef_construct=self.arm.get('ef_construction', 512)
                ),
            )
        self.wait_green()

    def wait_green(self) -> None:
        from qdrant_client import models as m

        time.sleep(1)
        while True:
            info = self.client.get_collection(COLLECTION)
            if (
                info.status == m.CollectionStatus.GREEN
                and info.optimizer_status == m.OptimizersStatusOneOf.OK
            ):
                return
            time.sleep(2)

    def to_filter(self, flt: FilterSpec) -> Any:
        from qdrant_client import models as m

        must: list[Any] = [m.FieldCondition(key=f, match=m.MatchValue(value=v)) for f, v in flt.eq]
        must += [m.FieldCondition(key=f, match=m.MatchValue(value=False)) for f in flt.is_false]
        if flt.sel_lt is not None:
            must.append(m.FieldCondition(key='sel', range=m.Range(lt=flt.sel_lt)))
        return m.Filter(must=must) if must else None

    def search(
        self, vector: np.ndarray, k: int, flt: FilterSpec, params: dict[str, Any]
    ) -> list[int]:
        from qdrant_client import models as m

        quant = None
        if self.arm.get('quantization'):
            quant = m.QuantizationSearchParams(
                rescore=True, oversampling=params.get('oversampling', 2.0)
            )
        resp = self.client.query_points(
            COLLECTION,
            query=vector.tolist(),
            query_filter=self.to_filter(flt),
            limit=k,
            search_params=m.SearchParams(hnsw_ef=params['hnsw_ef'], quantization=quant),
            with_payload=False,
            with_vectors=False,
        )
        return [int(p.id) for p in resp.points]

    def update_field(self, ids: np.ndarray, field: str, values: np.ndarray, threads: int) -> None:
        from qdrant_client import models as m

        groups: dict[int, list[int]] = {}
        for row_id, value in zip(ids, values, strict=True):
            groups.setdefault(int(value), []).append(int(row_id))
        items = list(groups.items())

        def work(a: int, b: int) -> None:
            for value, rows in items[a:b]:
                self.client.set_payload(COLLECTION, payload={field: value}, points=rows, wait=True)

        run_threads(work, len(items), 8, threads)
        self.wait_green()
        _ = m

    def delete_where(self, flt: FilterSpec) -> None:
        from qdrant_client import models as m

        self.client.delete(
            COLLECTION, points_selector=m.FilterSelector(filter=self.to_filter(flt)), wait=True
        )

    def settle(self) -> None:
        self.wait_green()

    def bytes_on_disk(self) -> int:
        import requests

        host, _, port = self.endpoint.partition(':')
        telemetry = requests.get(
            f'http://{host}:{port}/telemetry?details_level=4', timeout=60
        ).json()
        shards = telemetry['result']['collections']['collections'][0]['shards']
        total = 0
        for shard in shards:
            for seg in shard.get('local', {}).get('segments', []):
                total += int(seg['info'].get('disk_usage_bytes', 0))
        return total

    def count(self) -> int:
        return int(self.client.count(COLLECTION, exact=True).count)

    def drop(self) -> None:
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
