"""LanceDB adapter (embedded): IVF_PQ and IVF_HNSW_SQ vector indexes, scalar indexes on filters."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench.vectordb.adapters.base import Adapter, Meta


if TYPE_CHECKING:
    from scripts.bench.vectordb.vdb_lib import FilterSpec


TABLE = 'vdb_bench'
COLUMNS = (
    'project',
    'class_name',
    'class_validated',
    'test_holdout',
    'review_dismissed',
    'cluster_id',
    'sel',
)
BITMAP = ('class_validated', 'test_holdout', 'review_dismissed', 'project', 'class_name')


def where(flt: FilterSpec) -> str | None:
    parts = [f"{f} = '{v}'" if isinstance(v, str) else f'{f} = {int(v)}' for f, v in flt.eq]
    parts += [f'{f} = false' for f in flt.is_false]
    if flt.sel_lt is not None:
        parts.append(f'sel < {flt.sel_lt}')
    return ' AND '.join(parts) or None


class LanceAdapter(Adapter):
    engine = 'lancedb'

    def __init__(self, arm: dict[str, Any], endpoint: str) -> None:
        super().__init__(arm, endpoint)
        import lancedb

        self.path = Path(endpoint)
        self.db = lancedb.connect(str(self.path))
        self.table: Any = None

    def create(self, dim: int) -> None:
        import pyarrow as pa

        self.dim = dim
        if TABLE in self.db.table_names():
            self.db.drop_table(TABLE)
        fields = [pa.field('id', pa.int64()), pa.field('vec', pa.list_(pa.float32(), dim))]
        types = {
            'class_name': pa.string(),
            'class_validated': pa.bool_(),
            'test_holdout': pa.bool_(),
            'review_dismissed': pa.bool_(),
        }
        fields += [pa.field(f, types.get(f, pa.int64())) for f in COLUMNS]
        self.table = self.db.create_table(TABLE, schema=pa.schema(fields))

    def encode(self, vectors: np.ndarray, meta: Meta, ids: np.ndarray) -> Any:
        import pyarrow as pa

        cols: dict[str, Any] = {
            'id': pa.array(ids.astype(np.int64)),
            'vec': pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)), self.dim),
        }
        for f in COLUMNS:
            cols[f] = pa.array(meta[f])
        return pa.table(cols, schema=self.table.schema)

    def send(self, payload: Any) -> None:
        self.table.add(payload)

    def finish_load(self) -> None:
        index = self.arm['index']
        parts = self.arm.get('partitions') or max(1, round(self.table.count_rows() ** 0.5))
        if index == 'IVF_PQ':
            self.table.create_index(
                metric='cosine',
                vector_column_name='vec',
                index_type='IVF_PQ',
                num_partitions=parts,
                num_sub_vectors=self.arm.get('sub_vectors', 128),
            )
        elif index == 'IVF_HNSW_SQ':
            self.table.create_index(
                metric='cosine',
                vector_column_name='vec',
                index_type='IVF_HNSW_SQ',
                num_partitions=parts,
                m=self.arm.get('m', 16),
                ef_construction=self.arm.get('ef_construction', 512),
            )
        for f in COLUMNS:
            kind = 'BITMAP' if f in BITMAP else 'BTREE'
            self.table.create_scalar_index(f, index_type=kind)
        self.table.optimize()

    def search(
        self, vector: np.ndarray, k: int, flt: FilterSpec, params: dict[str, Any]
    ) -> list[int]:
        q = self.table.search(vector, vector_column_name='vec').metric('cosine')
        expr = where(flt)
        if expr:
            q = q.where(expr, prefilter=True)
        if self.arm['index'] != 'FLAT':
            q = q.nprobes(params['nprobes'])
            if params.get('refine'):
                q = q.refine_factor(params['refine'])
            if params.get('ef'):
                q = q.ef(max(params['ef'], k))
        rows = q.select(['id']).limit(k).to_arrow()
        return [int(x) for x in rows['id'].to_pylist()]

    def update_field(self, ids: np.ndarray, field: str, values: np.ndarray, threads: int) -> None:  # noqa: ARG002  # interface method; this implementation does not need the argument
        import pyarrow as pa

        data = pa.table(
            {'id': pa.array(ids.astype(np.int64)), field: pa.array(values.astype(np.int64))}
        )
        self.table.merge_insert('id').when_matched_update_all().execute(data)

    def delete_where(self, flt: FilterSpec) -> None:
        self.table.delete(where(flt))

    def settle(self) -> None:
        self.table.optimize(cleanup_older_than=__import__('datetime').timedelta(seconds=0))

    def bytes_on_disk(self) -> int:
        out = subprocess.run(  # nosec B603 B607  fixed argv list, no shell; du from PATH
            ['du', '-sb', str(self.path)], capture_output=True, text=True, check=True
        )
        return int(out.stdout.split()[0])

    def count(self) -> int:
        return int(self.table.count_rows())

    def drop(self) -> None:
        if TABLE in self.db.table_names():
            self.db.drop_table(TABLE)
