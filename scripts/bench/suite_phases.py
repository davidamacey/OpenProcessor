"""Triton, endpoint and VLM measurement phases of the baseline suite."""

from __future__ import annotations

import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

from scripts.bench.baseline_report import metrics_delta, percentile
from scripts.bench.suite_lib import best_point, parse_perf_csv, summarize_runs, triton_stats_delta
from scripts.bench.suite_probes import Gauges, chunks, run_cmd, scrape, triton_snapshot
from scripts.curation._project_worker_utils import curation_api_prefix, scoped_url
from scripts.datasets.bench_set import load_pin
from src.config.settings import TritonModelConfig


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
PERF_MODELS = (
    TritonModelConfig.YOLO_MODEL,
    TritonModelConfig.PE_IMAGE_MODEL,
    TritonModelConfig.FACE_DETECT_MODEL,
    TritonModelConfig.ARCFACE_MODEL,
    TritonModelConfig.CLIP_IMAGE_MODEL,
)
PERF_BATCHES = (1, 8, 16, 32)
ENDPOINTS = (
    ('detect', '/detect'),
    ('embed_image', '/embed/image'),
    ('faces_detect', '/faces/detect'),
    ('ocr_predict', '/ocr/predict'),
)
SDK_IMAGE = 'nvcr.io/nvidia/tritonserver:26.06-py3-sdk'


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
                runs: list[dict[str, Any]] = []
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
    base = scoped_url(args.api_url, curation_api_prefix(), slug, '')
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
