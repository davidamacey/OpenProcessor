"""Drive the vector store benchmark: one JSON document per (size, arm).

Start one engine at a time under its own compose project, then run its arms::

    docker compose -p vecdb -f docker/bench/vectordb/compose.yml up -d opensearch
    venv/bin/python scripts/bench/vectordb/run.py --dataset DIR --size 100000 \\
        --arms os_shipped,os_efc128 --endpoint 127.0.0.1:4701 --out docs/benchmarks/vectordb

Phases per arm: load (8 client threads), footprint, query matrix, metadata rewrite, streaming
insert, delete. ``--phases load1`` instead measures a single-client load only. Host load is
recorded before and after every phase; repeat runs decide what is noise.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.vectordb import dataset, hostinfo, phases_query, phases_write
from scripts.bench.vectordb.arms import ARMS


if TYPE_CHECKING:
    from scripts.bench.vectordb.adapters.base import Adapter


ADAPTERS = {
    'opensearch': ('scripts.bench.vectordb.adapters.opensearch', 'OpenSearchAdapter'),
    'milvus': ('scripts.bench.vectordb.adapters.milvus', 'MilvusAdapter'),
    'qdrant': ('scripts.bench.vectordb.adapters.qdrant', 'QdrantAdapter'),
    'pgvector': ('scripts.bench.vectordb.adapters.pgvector', 'PgvectorAdapter'),
    'lancedb': ('scripts.bench.vectordb.adapters.lancedb', 'LanceAdapter'),
    'faiss_flat': ('scripts.bench.vectordb.adapters.faiss_flat', 'FaissFlatAdapter'),
}
ALL_PHASES = ('load', 'footprint', 'query', 'rewrite', 'stream', 'delete')


def make_adapter(arm_name: str, endpoint: str) -> Adapter:
    arm = ARMS[arm_name]
    module, cls = ADAPTERS[arm['store']]
    result: Adapter = getattr(importlib.import_module(module), cls)(arm, endpoint)
    return result


def timed(name: str, fn: Any, doc: dict[str, Any]) -> Any:
    """Run ``fn`` and store its result under ``name`` with host load before and after."""
    before = hostinfo.host_snapshot()
    t0 = time.perf_counter()
    result = fn()
    doc.setdefault('host', {})[name] = {
        'before': before,
        'after': hostinfo.host_snapshot(),
        'seconds': time.perf_counter() - t0,
    }
    doc[name] = result
    print(f'  {name}: {time.perf_counter() - t0:.1f}s', flush=True)
    return result


def run_arm(arm_name: str, ds: dataset.Dataset, args: argparse.Namespace) -> dict[str, Any]:
    adapter = make_adapter(arm_name, args.endpoint)
    arm = ARMS[arm_name]
    doc: dict[str, Any] = {
        'arm': arm_name,
        'config': arm,
        'size': ds.size,
        'dim': int(ds.vectors.shape[1]),
        'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'repetition': args.repetition,
    }
    if adapter.container:
        doc['image'] = hostinfo.image_digest(adapter.container)
    phases = args.phases.split(',')
    try:
        if 'load1' in phases:
            timed('load', lambda: phases_write.load_phase(adapter, ds, args.batch, 1), doc)
            return doc
        timed('load', lambda: phases_write.load_phase(adapter, ds, args.batch, args.threads), doc)
        if 'footprint' in phases:
            timed('footprint', lambda: phases_write.footprint(adapter, ds.size), doc)
        if 'query' in phases:
            timed('query', lambda: phases_query.query_phase(adapter, ds, args.reps), doc)
        if 'query' in phases:
            doc['engine_stats_after_queries'] = adapter.engine_stats()
        op = doc.get('query', {}).get('operating_points')
        if op and 'rewrite' in phases:
            timed('rewrite', lambda: phases_write.rewrite_phase(adapter, ds, op, args.threads), doc)
        if op and 'stream' in phases and arm['store'] != 'faiss_flat':
            timed('stream', lambda: phases_write.stream_phase(adapter, ds, op), doc)
        if 'delete' in phases and arm['store'] != 'faiss_flat':
            project0 = next(f for f in ds.filters if f.name == 'project')
            timed('delete', lambda: phases_write.delete_phase(adapter, project0), doc)
    finally:
        try:
            adapter.drop()
        finally:
            adapter.close()
    return doc


def write_arm(arm_name: str, ds: dataset.Dataset, args: argparse.Namespace) -> Path:
    """Run one arm and write ``<out>/<size>/<arm>[_load1]_r<N>.json``."""
    outdir: Path = args.out / str(ds.size)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'{arm_name} @ {ds.size}', flush=True)
    doc = run_arm(arm_name, ds, args)
    suffix = '_load1' if 'load1' in args.phases else ''
    path = outdir / f'{arm_name}{suffix}_r{args.repetition}.json'
    path.write_text(json.dumps(doc, indent=1, default=float))
    return path


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument('--dataset', type=Path, required=True, help='dir with base.npz and queries.npz')
    ap.add_argument('--size', type=int, required=True)
    ap.add_argument('--arms', required=True, help='comma separated names from arms.py')
    ap.add_argument('--out', type=Path, default=Path('docs/benchmarks/vectordb'))
    ap.add_argument('--phases', default=','.join(ALL_PHASES))
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--batch', type=int, default=1000)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--repetition', type=int, default=1, help='run number, part of the file name')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument('--endpoint', default='')
    ap.add_argument('--build', action='store_true', help='build the dataset from --export first')
    ap.add_argument('--export', type=Path, nargs='+')
    args = ap.parse_args()
    if args.build:
        dataset.build(args.export, args.dataset)
    ds = dataset.load(args.dataset, args.size)
    for arm_name in args.arms.split(','):
        write_arm(arm_name, ds, args)


if __name__ == '__main__':
    main()
