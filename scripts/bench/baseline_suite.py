"""One entry point for the Wave 0 baseline: environment, ingest, Triton, endpoints, VLM.

Writes (and merges into) one JSON document with a fixed schema, so a phase can be
re-run alone. Phases (``--phases``, comma list; default all but ``vlm``):

* ``env``: GPU model/driver/CUDA, CPU, RAM, git sha, container image digests,
  Triton and OpenSearch versions, dataset pin hash, foreign GPU load before the runs.
* ``ingest``: ``--reps`` interleaved repetitions of ingest through
  ``POST .../ingest/upload`` and ``POST .../ingest/batch`` on a fresh project each,
  with the stage timers, Triton statistics, wire bytes, CPU seconds, GPU samples,
  cluster training wall time and settled OpenSearch storage per image and per vector.
* ``triton``: perf_analyzer points per model at batch 1/8/16/32 (SDK container).
* ``endpoints``: /detect, /embed/image, /faces/detect, /ocr/predict, sequential and concurrent.
* ``vlm``: ``POST .../vlm/label_batch`` on crops of the last ingest project.

    baseline_suite.py --manifest scripts/datasets/manifests/coco_bench_2000.json \\
        --images data/samples/coco_bench_2000/images --out artifacts_local/bench/v050.json \\
        --project base045 --api-url http://127.0.0.1:4903 --triton-url http://127.0.0.1:4900 \\
        --opensearch-url http://127.0.0.1:4907 --source-map HOST_DIR=/data/source/coco_bench_2000

Nothing here reads image content into the report; paths and names are not recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.baseline_report import metrics_delta, percentile
from scripts.bench.run_baseline import API_METRIC_PREFIXES, map_path
from scripts.bench.suite_lib import (
    SCHEMA,
    best_point,
    parse_perf_csv,
    storage_summary,
    summarize_runs,
    triton_stats_delta,
    triton_wire_bytes,
)
from scripts.bench.suite_probes import (
    Gauges,
    cgroup_cpu_s,
    chunks,
    phase_env,
    run_cmd,
    scrape,
    stage_table,
    triton_configs,
    triton_snapshot,
    wait_idle,
)
from scripts.datasets.bench_set import load_pin


VLM_CLASSES = (
    'person',
    'car',
    'truck',
    'bus',
    'motorcycle',
    'bicycle',
    'dog',
    'cat',
    'chair',
    'bottle',
)
PHASES = ('env', 'ingest', 'triton', 'endpoints', 'vlm')
DEFAULT_PHASES = ('env', 'ingest', 'triton', 'endpoints')
PERF_MODELS = (
    'yolov11_small_trt_end2end',
    'pe_image_encoder',
    'scrfd_10g_bnkps',
    'arcface_w600k_r50',
    'mobileclip2_s2_image_encoder',
)
PERF_BATCHES = (1, 8, 16, 32)
ENDPOINTS = (
    ('detect', '/detect'),
    ('embed_image', '/embed/image'),
    ('faces_detect', '/faces/detect'),
    ('ocr_predict', '/ocr/predict'),
)
SDK_IMAGE = 'nvcr.io/nvidia/tritonserver:26.06-py3-sdk'


# ---------------------------------------------------------------- ingest


class Wire:
    """Request and response body bytes seen by the HTTP client."""

    def __init__(self) -> None:
        self.sent = 0
        self.received = 0
        self._lock = threading.Lock()

    def on_request(self, request: httpx.Request) -> None:
        with self._lock:
            self.sent += int(request.headers.get('content-length', 0))

    def on_response(self, response: httpx.Response) -> None:
        response.read()
        with self._lock:
            self.received += len(response.content)


def post_batch(
    client: httpx.Client, base: str, mode: str, files: list[Path], source_map: list[str]
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    if mode == 'upload':
        handles = [(p.name, p.read_bytes()) for p in files]
        resp = client.post(
            f'{base}/ingest/upload', files=[('images', (n, b, 'image/jpeg')) for n, b in handles]
        )
    else:
        items = [{'path': map_path(str(p), source_map), 'source': 'baseline'} for p in files]
        resp = client.post(f'{base}/ingest/batch', json={'items': items})
    resp.raise_for_status()
    return resp.json()['summary'], time.monotonic() - started


def settle_and_count(client: httpx.Client, os_url: str, indexes: list[str]) -> dict[str, Any]:
    target = ','.join(sorted(set(indexes)))
    rows: list[dict[str, Any]] = []
    previous = -1
    settled_after = 0
    # Background writers (cluster refresh) keep rewriting documents for a while after the
    # last request; merge until nothing is deleted and the size repeats.
    for attempt in range(1, 13):
        settled_after = attempt
        client.post(
            f'{os_url}/{target}/_forcemerge', params={'max_num_segments': '1'}
        ).raise_for_status()
        client.post(f'{os_url}/{target}/_refresh').raise_for_status()
        client.post(f'{os_url}/{target}/_flush').raise_for_status()
        time.sleep(5)
        rows = (
            client.get(
                f'{os_url}/_cat/indices/{target}',
                params={
                    'format': 'json',
                    'bytes': 'b',
                    'h': 'index,docs.count,docs.deleted,pri.store.size',
                },
            )
            .raise_for_status()
            .json()
        )
        total = sum(int(r['pri.store.size']) for r in rows)
        if total == previous and all(int(r['docs.deleted']) == 0 for r in rows):
            break
        previous = total
    vectors, images = 0, 0
    for index in sorted(set(indexes)):
        mapping = (
            client.get(f'{os_url}/{index}/_mapping')
            .raise_for_status()
            .json()[index]['mappings']['properties']
        )
        for field, spec in _knn_fields(mapping):
            if spec['nested']:
                body = {
                    'size': 0,
                    'aggs': {
                        'n': {
                            'nested': {'path': spec['nested']},
                            'aggs': {'c': {'filter': {'exists': {'field': field}}}},
                        }
                    },
                }
                res = client.post(f'{os_url}/{index}/_search', json=body).raise_for_status().json()
                vectors += int(res['aggregations']['n']['c']['doc_count'])
            else:
                res = (
                    client.post(
                        f'{os_url}/{index}/_count', json={'query': {'exists': {'field': field}}}
                    )
                    .raise_for_status()
                    .json()
                )
                vectors += int(res['count'])
        if index.endswith('__images'):
            images = int(client.get(f'{os_url}/{index}/_count').raise_for_status().json()['count'])
    return {'rows': rows, 'vectors': vectors, 'images': images, 'merge_rounds': settled_after}


def _knn_fields(
    props: dict[str, Any], prefix: str = '', nested: str = ''
) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for name, spec in props.items():
        path = f'{prefix}{name}'
        if spec.get('type') == 'knn_vector':
            out.append((path, {'nested': nested}))
        elif 'properties' in spec:
            out.extend(
                _knn_fields(
                    spec['properties'], f'{path}.', path if spec.get('type') == 'nested' else nested
                )
            )
    return out


def run_autolabel(
    client: httpx.Client, base: str, params: dict[str, str], timeout: float = 3600.0
) -> dict[str, Any]:
    started = time.monotonic()
    # The cluster-refresh worker may hold the single auto_label slot; wait for it, then start ours.
    while True:
        resp = client.post(f'{base}/pipeline/auto_label/start', params=params)
        if resp.status_code != 409:
            break
        if time.monotonic() - started > timeout:
            raise TimeoutError('auto_label slot stayed busy')
        time.sleep(5)
    job = resp.raise_for_status().json()
    started = time.monotonic()
    while True:
        state = (
            client.get(f'{base}/pipeline/auto_label/status/{job["job_id"]}')
            .raise_for_status()
            .json()
        )
        if state['status'] in ('completed', 'failed', 'cancelled'):
            return {
                'status': state['status'],
                'wall_s': time.monotonic() - started,
                'stage_durations': state.get('stage_durations') or {},
            }
        if time.monotonic() - started > timeout:
            raise TimeoutError('auto_label did not finish')
        time.sleep(2)


def ingest_rep(
    args: argparse.Namespace,
    mode: str,
    rep: int,
    files: list[Path],
    warmup: bool = False,
    keep: bool = False,
) -> dict[str, Any]:
    idle = wait_idle(args.project)
    wire = Wire()
    slug = f'{args.prefix}-{mode}-{"warm" if warmup else f"r{rep}"}-{datetime.now(UTC):%H%M%S}'
    base = f'{args.api_url}/curation/projects/{slug}'
    hooks = {'request': [wire.on_request], 'response': [wire.on_response]}
    with httpx.Client(timeout=httpx.Timeout(3600.0), event_hooks=hooks) as client:
        client.post(
            f'{args.api_url}/curation/projects', json={'slug': slug, 'display_name': slug}
        ).raise_for_status()
        batches = chunks(files, args.batch_size)
        before_t = triton_snapshot(client, args.triton_url)
        before_m = scrape(client, f'{args.api_url}/metrics')
        cpu0 = {c: cgroup_cpu_s(f'{args.project}-{c}') for c in ('api', 'triton')}
        started = time.monotonic()
        with Gauges(args.project, args.gpu) as gauges, ThreadPoolExecutor(args.concurrency) as pool:
            results = list(
                pool.map(
                    lambda b: post_batch(client, base, mode, list(b), args.source_map), batches
                )
            )
        wall = time.monotonic() - started
        cpu = {c: cgroup_cpu_s(f'{args.project}-{c}') - cpu0[c] for c in cpu0}
        after_t = triton_snapshot(client, args.triton_url)
        after_m = scrape(client, f'{args.api_url}/metrics')
        if warmup:
            client.delete(base, params={'confirm': slug}).raise_for_status()
            return {'slug': slug}
        delta = triton_stats_delta(before_t, after_t)
        stages = stage_table(metrics_delta(before_m, after_m, API_METRIC_PREFIXES))
        ok = sum(s['successful'] for s, _ in results)
        failed = sum(s['failed'] for s, _ in results)
        if ok == 0:
            raise RuntimeError(f'{mode}: every image failed; check the paths and --source-map')
        lat = [t for _, t in results]
        cluster = run_autolabel(client, base, {'train_clusters': 'true', 'run_vlm': 'false'})
        names = list(client.get(base).raise_for_status().json()['resources']['indexes'].values())
        store = settle_and_count(client, args.opensearch_url.rstrip('/'), names)
        configs = triton_configs(client, args.triton_url, list(delta))
        if not keep:
            client.delete(base, params={'confirm': slug}).raise_for_status()
    summary = storage_summary(store['rows'], images=store['images'], vectors=store['vectors'])
    flat = {
        'images_per_s': ok / wall,
        'wall_s': wall,
        'failed': float(failed),
        'duplicates': float(sum(s['duplicates'] for s, _ in results)),
        'items_per_image': sum(s['crops_indexed'] for s, _ in results) / max(ok, 1),
        'request_p50_s': statistics.median(lat),
        'request_p95_s': percentile(lat, 95),
        'api_cpu_s_per_image': cpu['api'] / max(ok, 1),
        'triton_cpu_s_per_image': cpu['triton'] / max(ok, 1),
        'client_sent_bytes_per_image': wire.sent / max(ok, 1),
        'client_received_bytes_per_image': wire.received / max(ok, 1),
        'cluster_train_wall_s': cluster['wall_s'],
        'idle_api_cores_before': idle['cores']['api'],
        'bytes_per_image': summary['bytes_per_image'],
        'bytes_per_vector': summary['bytes_per_vector'],
    }
    for stage, row in stages.items():
        flat[f'stage_{stage}_mean_ms'] = row.get('mean_ms', 0.0)
        flat[f'stage_{stage}_seconds'] = row.get('seconds', 0.0)
    return {
        'mode': mode,
        'rep': rep,
        'slug': slug,
        'flat': flat,
        'stages': stages,
        'triton': delta,
        'triton_wire_bytes': triton_wire_bytes(delta, configs),
        'cluster': cluster,
        'storage': {**summary, 'merge_rounds': store['merge_rounds']},
        **gauges.result(),
    }


def phase_ingest(args: argparse.Namespace) -> dict[str, Any]:
    pin = load_pin(args.manifest)
    files = [args.images / r['file_name'] for r in pin['images']][: args.limit or None]
    modes = args.modes.split(',')
    for mode in modes:
        ingest_rep(args, mode, 0, files[: args.warmup], warmup=True)
    runs: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    for rep in range(1, args.reps + 1):
        for mode in modes:
            runs[mode].append(
                ingest_rep(args, mode, rep, files, keep=(mode == 'upload' and rep == args.reps))
            )
            args.out.with_suffix('.partial.json').write_text(json.dumps(runs, sort_keys=True))
    return {
        'config': {
            'images': len(files),
            'batch_size': args.batch_size,
            'concurrency': args.concurrency,
            'reps': args.reps,
            'warmup_images': args.warmup,
            'policy': 'all (detector + whole-image and per-detection embeddings)',
        },
        **{
            m: {'summary': summarize_runs([r['flat'] for r in rs]), 'runs': rs}
            for m, rs in runs.items()
        },
    }


# ---------------------------------------------------------------- triton


def phase_triton(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.perf_dir
    outdir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for model in args.perf_models.split(','):
        for batch in (int(b) for b in args.perf_batches.split(',')):
            csv_name = f'{model}_b{batch}.csv'
            run_cmd(
                [
                    'docker',
                    'run',
                    '--rm',
                    '--name',
                    f'{args.project}-pa-{model}-{batch}',
                    '--user',
                    f'{os.getuid()}:{os.getgid()}',
                    '--network',
                    f'{args.project}_triton_net',
                    '-v',
                    f'{outdir}:/out',
                    SDK_IMAGE,
                    'perf_analyzer',
                    '-m',
                    model,
                    '-u',
                    f'{args.project}-triton:8001',
                    '-i',
                    'grpc',
                    '-b',
                    str(batch),
                    '--concurrency-range',
                    '1:16:5',
                    '--measurement-interval',
                    '5000',
                    '--stability-percentage',
                    '10',
                    '-f',
                    f'/out/{csv_name}',
                ],
                timeout=900,
            )
            path = outdir / csv_name
            rows = parse_perf_csv(path.read_text()) if path.is_file() else []
            result.setdefault(model, {})[str(batch)] = {
                'points': rows,
                'best': best_point(rows) if rows else None,
            }
    return result


# ---------------------------------------------------------------- endpoints


def phase_endpoints(args: argparse.Namespace) -> dict[str, Any]:
    pin = load_pin(args.manifest)
    files = [args.images / r['file_name'] for r in pin['images']][: args.endpoint_images]
    blobs = [p.read_bytes() for p in files]
    out: dict[str, Any] = {'images': len(blobs)}
    with httpx.Client(timeout=120.0) as client:
        for name, path in ENDPOINTS:
            for threads in (1, args.concurrency):
                runs = []
                for _ in range(args.reps):
                    before = triton_snapshot(client, args.triton_url)
                    lat: list[float] = []

                    def one(blob: bytes, path: str = path, lat: list[float] = lat) -> None:
                        t = time.monotonic()
                        client.post(
                            f'{args.api_url}{path}', files={'image': ('i.jpg', blob, 'image/jpeg')}
                        ).raise_for_status()
                        lat.append(time.monotonic() - t)

                    started = time.monotonic()
                    with ThreadPoolExecutor(threads) as pool:
                        list(pool.map(one, blobs))
                    wall = time.monotonic() - started
                    delta = triton_stats_delta(before, triton_snapshot(client, args.triton_url))
                    runs.append(
                        {
                            'flat': {
                                'images_per_s': len(blobs) / wall,
                                'p50_ms': statistics.median(lat) * 1000,
                                'p95_ms': percentile(lat, 95) * 1000,
                            },
                            'triton': delta,
                        }
                    )
                out.setdefault(name, {})[f'threads_{threads}'] = {
                    'summary': summarize_runs([r['flat'] for r in runs]),
                    'runs': runs,
                }
    return out


# ---------------------------------------------------------------- vlm


def phase_vlm(args: argparse.Namespace, prior: dict[str, Any]) -> dict[str, Any]:
    slug = args.vlm_project or prior['ingest']['upload']['runs'][-1]['slug']
    base = f'{args.api_url}/curation/projects/{slug}'
    os_url = args.opensearch_url.rstrip('/')
    with httpx.Client(timeout=httpx.Timeout(3600.0)) as client:
        index = next(
            i
            for i in client.get(base).json()['resources']['indexes'].values()
            if i.endswith('__items')
        )
        hits = client.post(
            f'{os_url}/{index}/_search',
            json={
                'size': args.vlm_crops * args.vlm_reps + 8,
                '_source': False,
                'query': {'match_all': {}},
            },
        ).json()['hits']['hits']
        ids = [h['_id'] for h in hits]
        n = args.vlm_crops
        warm, pool_ids = ids[n * args.vlm_reps :], ids[: n * args.vlm_reps]
        for name in VLM_CLASSES:
            client.post(f'{base}/classes', json={'name': name, 'group': 'bench'})
        client.post(
            f'{base}/vlm/label_batch', json={'crop_ids': warm}, params=args.vlm_params
        ).raise_for_status()
        runs = []
        for rep in range(args.vlm_reps):
            ids = pool_ids[rep * n : (rep + 1) * n]  # fresh crops each rep: no cache hits
            before = scrape(client, f'{args.api_url}/metrics')
            with Gauges(args.project, args.vlm_gpu or args.gpu) as gauges:
                started = time.monotonic()
                errors = 0

                def label(batch: Sequence[str]) -> None:
                    nonlocal errors
                    r = client.post(
                        f'{base}/vlm/label_batch',
                        json={'crop_ids': list(batch)},
                        params=args.vlm_params,
                    )
                    if r.status_code != 200:
                        errors += 1

                with ThreadPoolExecutor(args.vlm_threads) as pool:
                    list(pool.map(label, chunks(ids, args.vlm_batch)))
                wall = time.monotonic() - started
            if errors:
                raise RuntimeError(f'{errors} label_batch requests failed')
            runs.append(
                {
                    'flat': {
                        'crops_per_s': len(ids) / wall,
                        'wall_s': wall,
                        'errors': float(errors),
                    },
                    **gauges.result(),
                    'metrics': metrics_delta(
                        before, scrape(client, f'{args.api_url}/metrics'), ('op_vlm',)
                    ),
                }
            )
    return {
        'crops': len(ids),
        'batch': args.vlm_batch,
        'threads': args.vlm_threads,
        'summary': summarize_runs([r['flat'] for r in runs]),
        'runs': runs,
    }


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--manifest', type=Path, required=True, help='bench_set pin (scripts/datasets/manifests)'
    )
    p.add_argument('--images', type=Path, required=True, help='directory holding the pinned images')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--project', required=True, help='compose project name (container prefix)')
    p.add_argument('--api-url', required=True)
    p.add_argument('--triton-url', required=True)
    p.add_argument('--opensearch-url', required=True)
    p.add_argument(
        '--phases', default=','.join(DEFAULT_PHASES), help=f'comma list from {",".join(PHASES)}'
    )
    p.add_argument('--modes', default='upload,batch')
    p.add_argument(
        '--source-map',
        action='append',
        default=[],
        metavar='HOST=CONTAINER',
        help='rewrite image paths for /ingest/batch',
    )
    p.add_argument('--prefix', default='bl')
    p.add_argument('--gpu', default='0')
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument(
        '--limit', type=int, default=0, help='use only the first N pinned images (smoke runs)'
    )
    p.add_argument('--idle-samples', type=int, default=30)
    p.add_argument('--perf-dir', type=Path, default=Path('artifacts_local/bench/perf_analyzer'))
    p.add_argument('--perf-models', default=','.join(PERF_MODELS))
    p.add_argument('--perf-batches', default=','.join(str(b) for b in PERF_BATCHES))
    p.add_argument('--endpoint-images', type=int, default=200)
    p.add_argument('--vlm-crops', type=int, default=200)
    p.add_argument('--vlm-reps', type=int, default=2)
    p.add_argument('--vlm-batch', type=int, default=32)
    p.add_argument('--vlm-threads', type=int, default=8)
    p.add_argument('--vlm-gpu')
    p.add_argument(
        '--vlm-project', help='project whose crops are labeled (default: last upload rep)'
    )
    p.add_argument(
        '--vlm-param',
        action='append',
        default=[],
        metavar='K=V',
        help='query param for label_batch',
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.vlm_params = dict(kv.split('=', 1) for kv in args.vlm_param)
    args.perf_dir = args.perf_dir.resolve()
    args.images = args.images.resolve()
    phases = args.phases.split(',')
    if set(phases) - set(PHASES):
        sys.exit(f'--phases must be from {PHASES}')
    doc: dict[str, Any] = (
        json.loads(args.out.read_text()) if args.out.is_file() else {'schema': SCHEMA}
    )
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        'ingest': lambda: phase_ingest(args),
        'triton': lambda: phase_triton(args),
        'endpoints': lambda: phase_endpoints(args),
        'vlm': lambda: phase_vlm(args, doc),
    }
    with httpx.Client(timeout=30.0) as client:
        for phase in phases:
            started = time.monotonic()
            result = phase_env(args, client) if phase == 'env' else runners[phase]()
            doc[phase] = {**doc.get(phase, {}), **result} if phase == 'triton' else result
            doc.setdefault('phase_seconds', {})[phase] = time.monotonic() - started
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
            print(f'phase {phase} done in {time.monotonic() - started:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
