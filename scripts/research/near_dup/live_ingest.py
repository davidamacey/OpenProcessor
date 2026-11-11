"""nd:P1 tier 2: push the constructed set through a live stack and read the vectors back.

Needs an isolated stack (project ``op-nd-t2``: Triton, API, OpenSearch; see the note, section 7). Subcommands:

* ``core``: ``POST /ingest/batch`` (the route of ``dedupe_r1.json``) in family order, originals of a chunk
  first, then read ``global_embedding`` and ``duplicate_group_id`` back from OpenSearch;
* ``pe``: the live GPU-decode model ``gpu_image_prep`` through Triton gRPC (``whole_embedding``, the vector
  the curation ingest stores as ``pe_embedding``), same order; files the model declines (status not ``ok``,
  for example PNG) are recorded with their status and left without a vector.

Both stop at ``--deadline-s`` after finishing the chunk in hand, so only complete families are kept.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

from scripts.research.near_dup import sweep


CHUNK_FAMILIES = 100
BATCH = 32


def chunk_order(rows: list[dict]) -> list[list[int]]:
    """Row indices per chunk of ``CHUNK_FAMILIES`` families: the originals first, then their copies."""
    n_fam = 1 + max(r['family'] for r in rows)
    chunks = []
    for lo in range(0, n_fam, CHUNK_FAMILIES):
        fams = range(lo, min(lo + CHUNK_FAMILIES, n_fam))
        idx = [i for i, r in enumerate(rows) if r['family'] in fams]
        chunks.append(sorted(idx, key=lambda i: (rows[i]['transform'] != 'orig', i)))
    return chunks


def ingest_core(api: str, rows: list[dict], deadline: float, out: Path) -> dict:
    import requests

    done: list[int] = []
    t0 = time.time()
    failed = 0
    for chunk in chunk_order(rows):
        if time.time() > deadline:
            break
        for s in range(0, len(chunk), BATCH):
            part = chunk[s : s + BATCH]
            files = [
                ('images', (Path(rows[i]['path']).name, Path(rows[i]['path']).read_bytes()))
                for i in part
            ]
            resp = requests.post(
                f'{api}/ingest/batch',
                files=files,
                data={
                    'image_ids': json.dumps([f'r{i}' for i in part]),
                    'skip_duplicates': 'false',
                    'detect_near_duplicates': 'true',
                },
                timeout=600,
            )
            if resp.status_code != 201:
                failed += len(part)
                sys.stderr.write(f'batch failed {resp.status_code} {resp.text[:200]}\n')
        done.extend(chunk)
    stats = {
        'ingested': len(done),
        'failed': failed,
        'seconds': time.time() - t0,
        'rows': sorted(done),
    }
    out.write_text(json.dumps(stats))
    return stats


def read_core(
    os_url: str, index: str, n_rows: int
) -> tuple[np.ndarray, np.ndarray, dict[int, str | None]]:
    import requests

    vec = np.zeros((n_rows, 512), dtype=np.float32)
    have = np.zeros(n_rows, dtype=bool)
    groups: dict[int, str | None] = {}
    body = {
        'size': 1000,
        'query': {'match_all': {}},
        '_source': ['image_id', 'global_embedding', 'duplicate_group_id'],
    }
    r = requests.post(f'{os_url}/{index}/_search?scroll=2m', json=body, timeout=120).json()
    while r['hits']['hits']:
        for h in r['hits']['hits']:
            i = int(h['_source']['image_id'][1:])
            vec[i] = h['_source']['global_embedding']
            have[i] = True
            groups[i] = h['_source'].get('duplicate_group_id')
        r = requests.post(
            f'{os_url}/_search/scroll',
            json={'scroll': '2m', 'scroll_id': r['_scroll_id']},
            timeout=120,
        ).json()
    return vec, have, groups


def ingest_pe(
    grpc_url: str, rows: list[dict], deadline: float
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    import tritonclient.grpc as grpcclient

    client = grpcclient.InferenceServerClient(grpc_url)
    client.load_model('gpu_image_prep')
    results: queue.Queue = queue.Queue()
    client.start_stream(callback=lambda result, error: results.put((result, error)))
    vec = np.zeros((len(rows), 1024), dtype=np.float32)
    have = np.zeros(len(rows), dtype=bool)
    statuses: dict[str, int] = {}
    for chunk in chunk_order(rows):
        if time.time() > deadline:
            break
        for s in range(0, len(chunk), 16):
            part = chunk[s : s + 16]
            for i in part:
                data = np.frombuffer(Path(rows[i]['path']).read_bytes(), dtype=np.uint8)
                inputs = [
                    grpcclient.InferInput('jpeg', list(data.shape), 'UINT8'),
                    grpcclient.InferInput('embed_crops', [1], 'BOOL'),
                ]
                inputs[0].set_data_from_numpy(data)
                inputs[1].set_data_from_numpy(np.array([False]))
                client.async_stream_infer('gpu_image_prep', inputs, request_id=str(i))
            for _ in part:
                res, err = results.get(timeout=300)
                if err is not None:
                    statuses['error'] = statuses.get('error', 0) + 1
                    continue
                i = int(res.get_response().id)
                st = res.as_numpy('status')[0]
                st = st.decode() if isinstance(st, bytes) else str(st)
                statuses[st] = statuses.get(st, 0) + 1
                if st == 'ok':
                    v = res.as_numpy('whole_embedding').astype(np.float32)
                    vec[i] = v / np.linalg.norm(v)
                    have[i] = True
    client.stop_stream()
    return vec, have, statuses


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=['core', 'pe'])
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='output prefix')
    ap.add_argument('--api', default='http://127.0.0.1:14603')
    ap.add_argument('--opensearch', default='http://127.0.0.1:14607')
    ap.add_argument('--index', default='visual_search_global')
    ap.add_argument('--grpc', default='127.0.0.1:14601')
    ap.add_argument(
        '--deadline-s',
        type=float,
        required=True,
        help='stop starting new chunks after this many seconds',
    )
    args = ap.parse_args(argv)
    rows = sweep.pool_from_manifest(json.loads(args.manifest.read_text()))
    deadline = time.time() + args.deadline_s
    t0 = time.time()
    if args.cmd == 'core':
        stats = ingest_core(args.api, rows, deadline, args.out.with_suffix('.ingest.json'))
        vec, have, groups = read_core(args.opensearch, args.index, len(rows))
        np.save(args.out.with_suffix('.npy'), vec)
        args.out.with_suffix('.meta.json').write_text(
            json.dumps(
                {
                    'have': np.flatnonzero(have).tolist(),
                    'groups': {str(k): v for k, v in groups.items()},
                    'ingest': {k: v for k, v in stats.items() if k != 'rows'},
                }
            )
        )
    else:
        vec, have, statuses = ingest_pe(args.grpc, rows, deadline)
        np.save(args.out.with_suffix('.npy'), vec)
        args.out.with_suffix('.meta.json').write_text(
            json.dumps(
                {
                    'have': np.flatnonzero(have).tolist(),
                    'statuses': statuses,
                    'seconds': time.time() - t0,
                }
            )
        )
    sys.stdout.write(f'{args.cmd}: {int(have.sum())} vectors in {time.time() - t0:.0f} s\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
