"""Speed of the ``embed`` reprocess scope (what embed-missing runs) on an ingested project.

Ingests a slice of the pinned set into a fresh project, then re-embeds every crop and frame
vector (``POST .../reprocess``, scope ``embed``, rewrite) ``--reps`` times and reports, per
repetition, the wall time, images/s, the PE engine batch histogram and request count from
Triton's statistics, and the foreign GPU share. One repetition is only comparable when the
foreign share is 0.

    embed_missing_bench.py --manifest scripts/datasets/manifests/coco_bench_2000.json \\
        --images data/samples/coco_bench_2000/images --out artifacts_local/bench/embed.json \\
        --project <compose project> --api-url http://127.0.0.1:<api> \\
        --triton-url http://127.0.0.1:<triton> --source-map <host dir>=<dir in the API container> \\
        --gpu <index> --count 500 --reps 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.baseline_suite import post_batch
from scripts.bench.suite_lib import triton_stats_delta
from scripts.bench.suite_probes import Gauges, cgroup_cpu_s, chunks, triton_snapshot
from scripts.curation._project_worker_utils import curation_api_prefix, scoped_url
from scripts.datasets.bench_set import load_pin


POLL_S = 1.0


def embed_run(client: httpx.Client, base: str, triton_url: str, gpu: str, project: str) -> dict:
    """One rewrite of every crop and frame vector; the wall time covers the whole job."""
    body = {
        'targets': {'filter': {'embedding_state': ['embedded']}},
        'scopes': ['embed'],
        'embed': {'only_missing': False, 'parts': ['crop', 'frame']},
        'dry_run': False,
    }
    before = triton_snapshot(client, triton_url)
    cpu0 = cgroup_cpu_s(f'{project}-api')
    with Gauges(project, gpu) as gauges:
        started = time.monotonic()
        reply = client.post(f'{base}/reprocess', json=body)
        if reply.is_error:
            raise RuntimeError(f'reprocess refused: {reply.status_code} {reply.text[:500]}')
        resp = reply.json()
        job = resp.get('job')
        while job and job['status'] in ('queued', 'running'):
            time.sleep(POLL_S)
            job = client.get(f'{base}/reprocess/jobs/{job["job_id"]}').raise_for_status().json()
        wall = time.monotonic() - started
    api_cpu = cgroup_cpu_s(f'{project}-api') - cpu0
    delta = triton_stats_delta(before, triton_snapshot(client, triton_url))
    done = (job or {}).get('images_done') or sum(s['queued'] for s in resp['scopes'])
    return {
        'wall_s': wall,
        'images': done,
        'images_per_s': done / wall,
        'api_cpu_s_per_image': api_cpu / max(done, 1),
        'api_cores': api_cpu / wall,
        'status': (job or {}).get('status', 'inline'),
        'failed': (job or {}).get('images_failed', 0),
        'pe': delta.get('pe_image_encoder', {}),
        **gauges.result(),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--project', required=True, help='compose project name')
    p.add_argument('--api-url', required=True)
    p.add_argument('--triton-url', required=True)
    p.add_argument('--source-map', action='append', default=[])
    p.add_argument('--gpu', default='0')
    p.add_argument('--count', type=int, default=500)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--concurrency', type=int, default=16)
    args = p.parse_args()

    pin = load_pin(args.manifest)
    files = [args.images / r['file_name'] for r in pin['images']][: args.count]
    slug = f'embedbench-{datetime.now(UTC):%H%M%S}'
    base = scoped_url(args.api_url, curation_api_prefix(), slug, '')
    runs: list[dict[str, Any]] = []
    with httpx.Client(timeout=httpx.Timeout(3600.0)) as client:
        client.post(
            f'{args.api_url}{curation_api_prefix()}/projects',
            json={'slug': slug, 'display_name': slug},
        ).raise_for_status()
        try:
            with ThreadPoolExecutor(args.concurrency) as pool:
                list(
                    pool.map(
                        lambda b: post_batch(client, base, 'batch', list(b), args.source_map),
                        chunks(files, args.batch_size),
                    )
                )
            runs.extend(
                embed_run(client, base, args.triton_url, args.gpu, args.project)
                for _ in range(args.reps)
            )
        finally:
            client.delete(base, params={'confirm': slug}).raise_for_status()
    rates = [r['images_per_s'] for r in runs]
    summary = {
        'images': len(files),
        'median_images_per_s': statistics.median(rates),
        'min_images_per_s': min(rates),
        'max_images_per_s': max(rates),
    }
    args.out.write_text(json.dumps({'summary': summary, 'runs': runs}, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
