"""Read every item vector of a project the old way and the one-reader way, and compare.

Arm ``old`` is the request the clustering snapshot used to send: one point-in-time cursor, page
5,000, the vector through ``_source`` (JSON float lists). Arm ``new`` is the product reader
(``read_vectors``: sliced point-in-time pages at once, ``docvalue_fields``, base64 float32).
Each arm reads the same pool (embedded, not validated, not confident-source, not excluded),
sorted by id; the report has rows, seconds, MB/s of float32 and whether the arrays are bitwise
identical. Only reads.

    .venv/bin/python scripts/bench/vector_read_probe.py --project SLUG [--reps 3] --out run.json

Run it inside the api container (or a venv that reaches OpenSearch) with the stack up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from src.config import get_settings
from src.config.curation_accessors import items_index
from src.db.engine import run_with_engine
from src.domain.embedding_spaces import Target
from src.services.curation.clustering.pool_snapshot import item_pool_query, snapshot_items
from src.services.embedding_spaces.resolve import embedding_space
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import abind_script_project, add_project_argument


OLD_PAGE = 5000
KEEP_ALIVE = '5m'


async def read_old(client: Any, field: str) -> tuple[list[str], np.ndarray]:
    """The superseded reader's request shape, kept here only as the comparison arm."""
    pit = (await client.create_pit(index=items_index(), keep_alive=KEEP_ALIVE))['pit_id']
    ids: list[str] = []
    rows: list[list[float]] = []
    after: list[Any] | None = None
    try:
        while True:
            body: dict[str, Any] = {
                'size': OLD_PAGE,
                'query': item_pool_query(field),
                '_source': [field],
                'pit': {'id': pit, 'keep_alive': KEEP_ALIVE},
                'sort': [{'_shard_doc': 'asc'}],
            }
            if after is not None:
                body['search_after'] = after
            hits = (await client.search(body=body))['hits']['hits']
            if not hits:
                break
            for hit in hits:
                vector = (hit.get('_source') or {}).get(field)
                if vector:
                    ids.append(hit['_id'])
                    rows.append(vector)
            after = hits[-1]['sort']
    finally:
        await client.delete_pit(body={'pit_id': [pit]})
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    order = sorted(range(len(ids)), key=ids.__getitem__)
    return [ids[i] for i in order], np.asarray(rows, dtype=np.float32)[order]


async def timed(label: str, read: Any) -> tuple[dict[str, Any], tuple[list[str], np.ndarray]]:
    t0 = time.perf_counter()
    ids, matrix = await read()
    secs = time.perf_counter() - t0
    mb = matrix.nbytes / 1e6
    return {
        'arm': label,
        'rows': len(ids),
        'dim': int(matrix.shape[1]) if matrix.size else 0,
        'seconds': round(secs, 3),
        'mb_per_s': round(mb / secs, 1) if secs else None,
    }, (ids, matrix)


async def run(project: str, reps: int, out: Path | None) -> int:
    await abind_script_project(project)
    field = (await embedding_space(Target.ITEM)).vector_field
    client = make_script_opensearch([get_settings().opensearch_url])
    results: list[dict[str, Any]] = []
    try:
        load_before = Path('/proc/loadavg').read_text().split()[0]
        for rep in range(reps):
            old_row, old = await timed('old', lambda: read_old(client, field))
            new_row, new = await timed('new', lambda: snapshot_items(client, field))
            same = (
                old[0] == new[0] and old[1].shape == new[1].shape and np.array_equal(old[1], new[1])
            )
            bitwise = same and old[1].tobytes() == new[1].tobytes()
            results.append(
                {
                    'rep': rep,
                    'old': old_row,
                    'new': new_row,
                    'identical': bool(same),
                    'bitwise': bool(bitwise),
                }
            )
            print(json.dumps(results[-1]))
            del old, new
        report = {
            'project': project,
            'field': field,
            'load1_before': load_before,
            'load1_after': Path('/proc/loadavg').read_text().split()[0],
            'reps': results,
        }
    finally:
        await client.close()
    if out:
        out.write_text(json.dumps(report, indent=2))
    return 0 if all(r['bitwise'] for r in results) else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_project_argument(p)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()
    return asyncio.run(
        run_with_engine(run(args.project, args.reps, args.out), 'op-vector-read-probe')
    )


if __name__ == '__main__':
    raise SystemExit(main())
