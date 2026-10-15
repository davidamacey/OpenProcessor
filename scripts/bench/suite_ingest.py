"""Ingest measurement of the baseline suite: the request wire counters, batch posting,
OpenSearch settle and count, auto-label, and one ingest repetition."""

from __future__ import annotations

import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from scripts.bench.baseline_report import metrics_delta, percentile
from scripts.bench.run_baseline import API_METRIC_PREFIXES, map_path
from scripts.bench.suite_lib import (
    storage_summary,
    summarize_runs,
    triton_stats_delta,
    triton_wire_bytes,
)
from scripts.bench.suite_probes import (
    Gauges,
    cgroup_cpu_s,
    chunks,
    scrape,
    stage_table,
    triton_configs,
    triton_snapshot,
    wait_idle,
)
from scripts.curation._project_worker_utils import curation_api_prefix, scoped_url
from scripts.datasets.bench_set import load_pin


if TYPE_CHECKING:
    import argparse
    from pathlib import Path


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


def set_embedding_mode(client: httpx.Client, base: str, mode: str) -> None:
    """Set a fresh project's ingest embedding mode (the default, ``all``, needs no call)."""
    if mode == 'all':
        return
    revision = client.get(f'{base}/ingest/policy').raise_for_status().json()['revision']
    body = {'embedding': {'mode': mode}, 'expected_revision': revision}
    client.put(f'{base}/ingest/policy', json=body).raise_for_status()


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
    base = scoped_url(args.api_url, curation_api_prefix(), slug, '')
    hooks: dict[str, list[Any]] = {'request': [wire.on_request], 'response': [wire.on_response]}
    with httpx.Client(timeout=httpx.Timeout(3600.0), event_hooks=hooks) as client:
        client.post(
            f'{args.api_url}{curation_api_prefix()}/projects',
            json={'slug': slug, 'display_name': slug},
        ).raise_for_status()
        set_embedding_mode(client, base, getattr(args, 'embedding_mode', 'all'))
        batches = chunks(files, args.batch_size)
        before_t = triton_snapshot(client, args.triton_url)
        before_m = scrape(client, f'{args.api_url}/metrics')
        cpu0 = {c: cgroup_cpu_s(f'{args.project}-{c}') for c in ('api', 'triton')}
        started = time.monotonic()
        with (
            getattr(args, 'gauges_cls', Gauges)(args.project, args.gpu) as gauges,
            ThreadPoolExecutor(args.concurrency) as pool,
        ):
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
        cluster = (
            {'status': 'skipped', 'wall_s': 0.0, 'stage_durations': {}}
            if getattr(args, 'skip_cluster', False)
            else run_autolabel(client, base, {'train_clusters': 'true', 'run_vlm': 'false'})
        )
        names = list(client.get(base).raise_for_status().json()['resources']['indexes'].values())
        store: dict[str, Any] = (
            {'rows': [], 'vectors': 0, 'images': 0, 'merge_rounds': 0}
            if getattr(args, 'skip_settle', False)
            else settle_and_count(client, args.opensearch_url.rstrip('/'), names)
        )
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
            'policy': (
                'all (detector + whole-image and per-detection embeddings)'
                if getattr(args, 'embedding_mode', 'all') == 'all'
                else 'lazy (detector + whole-image embedding; no per-detection embeddings)'
            ),
        },
        **{
            m: {'summary': summarize_runs([r['flat'] for r in rs]), 'runs': rs}
            for m, rs in runs.items()
        },
    }


# ---------------------------------------------------------------- triton
