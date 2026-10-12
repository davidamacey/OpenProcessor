"""Milvus standalone adapter: HNSW, IVF_FLAT and IVF_SQ8 with inverted scalar indexes."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from scripts.bench.vectordb.adapters.base import Adapter, Meta, run_threads


if TYPE_CHECKING:
    import numpy as np

    from scripts.bench.vectordb.vdb_lib import FilterSpec


COLLECTION = 'vdb_bench'
INT_FIELDS = ('project', 'cluster_id', 'sel')
BOOL_FIELDS = ('class_validated', 'test_holdout', 'review_dismissed')


def expression(flt: FilterSpec) -> str:
    parts = [f'{f} == {v!r}' if isinstance(v, str) else f'{f} == {int(v)}' for f, v in flt.eq]
    parts += [f'{f} == false' for f in flt.is_false]
    if flt.sel_lt is not None:
        parts.append(f'sel < {flt.sel_lt}')
    return ' and '.join(parts)


class MilvusAdapter(Adapter):
    engine = 'milvus'
    container = 'milvus'
    data_path = '/var/lib/milvus'

    def __init__(self, arm: dict[str, Any], endpoint: str) -> None:
        super().__init__(arm, endpoint)
        from pymilvus import MilvusClient

        self.client = MilvusClient(uri=f'http://{endpoint}', timeout=3600)

    def create(self, dim: int) -> None:
        from pymilvus import DataType

        if self.client.has_collection(COLLECTION):
            self.client.drop_collection(COLLECTION)
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field('id', DataType.INT64, is_primary=True)
        schema.add_field('vec', DataType.FLOAT_VECTOR, dim=dim)
        partition_key = bool(self.arm.get('partition_key'))
        schema.add_field('project', DataType.INT64, is_partition_key=partition_key)
        schema.add_field('class_name', DataType.VARCHAR, max_length=64)
        for f in BOOL_FIELDS:
            schema.add_field(f, DataType.BOOL)
        schema.add_field('cluster_id', DataType.INT64)
        schema.add_field('sel', DataType.INT64)
        kwargs: dict[str, Any] = {}
        if partition_key:
            kwargs['num_partitions'] = 16
        self.client.create_collection(COLLECTION, schema=schema, **kwargs)

    def index_params(self) -> Any:
        ip = self.client.prepare_index_params()
        index = self.arm['index']
        params: dict[str, Any] = {}
        if index == 'HNSW':
            params = {
                'M': self.arm.get('m', 16),
                'efConstruction': self.arm.get('ef_construction', 512),
            }
        else:
            params = {'nlist': self.arm.get('nlist', 1024)}
        ip.add_index('vec', index_type=index, metric_type='COSINE', params=params)
        for f in (*INT_FIELDS, 'class_name', *BOOL_FIELDS):
            if f == 'project' and self.arm.get('partition_key'):
                continue
            ip.add_index(f, index_type='INVERTED')
        return ip

    def encode(self, vectors: np.ndarray, meta: Meta, ids: np.ndarray) -> list[dict[str, Any]]:
        rows = []
        for pos in range(len(vectors)):
            row: dict[str, Any] = {'id': int(ids[pos]), 'vec': vectors[pos]}
            for f in (*INT_FIELDS, 'class_name', *BOOL_FIELDS):
                value = meta[f][pos]
                row[f] = value.item() if hasattr(value, 'item') else value
            rows.append(row)
        return rows

    def send(self, payload: Any) -> None:
        self.client.insert(COLLECTION, payload)

    def finish_load(self) -> None:
        self.client.flush(COLLECTION)
        self.client.create_index(COLLECTION, self.index_params(), timeout=7200)
        self.wait_indexed()
        self.client.load_collection(COLLECTION, timeout=3600)

    def wait_indexed(self) -> None:
        while True:
            info = self.client.describe_index(COLLECTION, 'vec')
            if int(info.get('pending_index_rows', 0)) == 0 and info.get('state') == 'Finished':
                return
            time.sleep(2)

    def search(
        self, vector: np.ndarray, k: int, flt: FilterSpec, params: dict[str, Any]
    ) -> list[int]:
        search_params: dict[str, Any] = {}
        if 'ef' in params:
            search_params['ef'] = max(params['ef'], k)
        if 'nprobe' in params:
            search_params['nprobe'] = params['nprobe']
        resp = self.client.search(
            COLLECTION,
            data=[vector],
            anns_field='vec',
            filter=expression(flt),
            limit=k,
            search_params={'metric_type': 'COSINE', 'params': search_params},
            output_fields=[],
        )
        return [int(h['id']) for h in resp[0]]

    def update_field(self, ids: np.ndarray, field: str, values: np.ndarray, threads: int) -> None:
        def work(a: int, b: int) -> None:
            rows = [{'id': int(ids[p]), field: int(values[p])} for p in range(a, b)]
            self.client.upsert(COLLECTION, rows, partial_update=True)

        run_threads(work, len(ids), 2000, threads)
        self.client.flush(COLLECTION)

    def delete_where(self, flt: FilterSpec) -> None:
        self.client.delete(COLLECTION, filter=expression(flt))
        self.client.flush(COLLECTION)

    def settle(self) -> None:
        self.client.compact(COLLECTION)
        time.sleep(30)

    def bytes_on_disk(self) -> int:
        return -1

    def count(self) -> int:
        return int(
            self.client.query(COLLECTION, filter='', output_fields=['count(*)'])[0]['count(*)']
        )

    def drop(self) -> None:
        if self.client.has_collection(COLLECTION):
            self.client.drop_collection(COLLECTION)

    def close(self) -> None:
        self.client.close()
