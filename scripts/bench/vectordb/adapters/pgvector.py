"""pgvector adapter: ``vector`` and ``halfvec`` columns, HNSW, B-tree filter indexes.

Rows are copied in first and the HNSW index is built afterwards (the fast path pgvector
documents for bulk loads); ``finish_load`` therefore includes the index build.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from scripts.bench.vectordb.adapters.base import Adapter, Meta, run_threads


if TYPE_CHECKING:
    import numpy as np

    from scripts.bench.vectordb.vdb_lib import FilterSpec


TABLE = 'items'
COLUMNS = (
    'project',
    'class_name',
    'class_validated',
    'test_holdout',
    'review_dismissed',
    'cluster_id',
    'sel',
)
COLUMN_TYPES = {
    'project': 'int',
    'class_name': 'text',
    'class_validated': 'boolean',
    'test_holdout': 'boolean',
    'review_dismissed': 'boolean',
    'cluster_id': 'int',
    'sel': 'int',
}


def where_clause(flt: FilterSpec) -> tuple[str, list[Any]]:
    parts: list[str] = []
    args: list[Any] = []
    for field, value in flt.eq:
        parts.append(f'{field} = %s')
        args.append(value if isinstance(value, str) else int(value))
    parts += [f'NOT {f}' for f in flt.is_false]
    if flt.sel_lt is not None:
        parts.append('sel < %s')
        args.append(flt.sel_lt)
    return (' WHERE ' + ' AND '.join(parts)) if parts else '', args


class PgvectorAdapter(Adapter):
    engine = 'pgvector'
    container = 'pgvector'
    data_path = '/var/lib/postgresql/data'

    def __init__(self, arm: dict[str, Any], endpoint: str) -> None:
        super().__init__(arm, endpoint)
        self.host, _, port = endpoint.partition(':')
        self.port = int(port)
        self.local = threading.local()
        self.vtype = 'halfvec' if arm.get('half') else 'vector'

    def connect(self, autocommit: bool = True) -> Any:
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(
            host=self.host,
            port=self.port,
            user='postgres',
            password='bench',
            dbname='bench',
            autocommit=autocommit,
        )
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(conn)
        return conn

    def conn(self) -> Any:
        """One connection per client thread, with the per-session search settings applied."""
        if not hasattr(self.local, 'c'):
            c = self.connect()
            c.execute('SET hnsw.iterative_scan = relaxed_order')
            c.execute('SET hnsw.max_scan_tuples = 200000')
            self.local.c = c
            self.local.ef = None
        return self.local.c

    def create(self, dim: int) -> None:
        with self.connect() as c:
            c.execute(f'DROP TABLE IF EXISTS {TABLE}')
            cols = ', '.join(f'{f} {t}' for f, t in COLUMN_TYPES.items())
            c.execute(
                f'CREATE TABLE {TABLE} (id bigint PRIMARY KEY, embedding {self.vtype}({dim}), {cols})'
            )
        self.dim = dim

    def encode(self, vectors: np.ndarray, meta: Meta, ids: np.ndarray) -> list[list[Any]]:
        rows = []
        for pos in range(len(vectors)):
            row: list[Any] = [int(ids[pos]), vectors[pos].tolist()]
            for f in COLUMNS:
                value = meta[f][pos]
                row.append(value.item() if hasattr(value, 'item') else value)
            rows.append(_fmt(row))
        return rows

    def send(self, payload: Any) -> None:
        names = ', '.join(('id', 'embedding', *COLUMNS))
        with (
            self.connect() as c,
            c.cursor() as cur,
            cur.copy(f'COPY {TABLE} ({names}) FROM STDIN') as cp,
        ):
            for row in payload:
                cp.write_row(row)

    def finish_load(self) -> None:
        ops = 'halfvec_cosine_ops' if self.vtype == 'halfvec' else 'vector_cosine_ops'
        with self.connect() as c:
            c.execute(
                f'CREATE INDEX hnsw_idx ON {TABLE} USING hnsw (embedding {ops}) '
                f'WITH (m = {self.arm.get("m", 16)}, ef_construction = {self.arm.get("ef_construction", 512)})'
            )
            for f in COLUMNS:
                c.execute(f'CREATE INDEX idx_{f} ON {TABLE} ({f})')
            c.execute(f'ANALYZE {TABLE}')

    def search(
        self, vector: np.ndarray, k: int, flt: FilterSpec, params: dict[str, Any]
    ) -> list[int]:
        c = self.conn()
        ef = int(params['ef_search'])
        if self.local.ef != ef:
            c.execute(f'SET hnsw.ef_search = {min(ef, 1000)}')
            self.local.ef = ef
        where, args = where_clause(flt)
        cast = 'halfvec' if self.vtype == 'halfvec' else 'vector'
        sql = f'SELECT id FROM {TABLE}{where} ORDER BY embedding <=> %s::{cast} LIMIT {int(k)}'
        rows = c.execute(sql, [*args, vector]).fetchall()
        return [int(r[0]) for r in rows]

    def update_field(self, ids: np.ndarray, field: str, values: np.ndarray, threads: int) -> None:
        def work(a: int, b: int) -> None:
            with self.connect() as c:
                c.execute(
                    f'UPDATE {TABLE} t SET {field} = v.val FROM (SELECT unnest(%s::bigint[]) AS id, '
                    f'unnest(%s::int[]) AS val) v WHERE t.id = v.id',
                    [[int(x) for x in ids[a:b]], [int(x) for x in values[a:b]]],
                )

        run_threads(work, len(ids), 2000, threads)

    def delete_where(self, flt: FilterSpec) -> None:
        where, args = where_clause(flt)
        with self.connect() as c:
            c.execute(f'DELETE FROM {TABLE}{where}', args)

    def settle(self) -> None:
        with self.connect() as c:
            c.execute(f'VACUUM (ANALYZE) {TABLE}')
            c.execute('CHECKPOINT')

    def bytes_on_disk(self) -> int:
        with self.connect() as c:
            row = c.execute(f"SELECT pg_total_relation_size('{TABLE}')").fetchone()
        return int(row[0])

    def count(self) -> int:
        with self.connect() as c:
            return int(c.execute(f'SELECT count(*) FROM {TABLE}').fetchone()[0])

    def drop(self) -> None:
        with self.connect() as c:
            c.execute(f'DROP TABLE IF EXISTS {TABLE}')


def _fmt(row: list[Any]) -> list[Any]:
    """COPY text form: the vector as ``[a,b,...]``, booleans as t/f."""
    out: list[Any] = []
    for x in row:
        if isinstance(x, list):
            out.append('[' + ','.join(f'{v:.7g}' for v in x) + ']')
        elif isinstance(x, bool):
            out.append('t' if x else 'f')
        else:
            out.append(x)
    return out
