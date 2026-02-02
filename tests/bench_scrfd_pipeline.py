#!/usr/bin/env python3
"""
SCRFD Face Recognition Pipeline Benchmark

Measures throughput (RPS) of the full face recognition pipeline:
  SCRFD detection -> Umeyama alignment -> ArcFace embedding

Tests with:
1. Serial single-client throughput
2. Concurrent multi-client throughput (simulates production load)
3. Batch API throughput

Uses face images from test_images/faces/lfw-deepfunneled/ for realistic workloads.

Reference performance targets (from InsightFace-REST on RTX 4090):
- SCRFD-10G detection: ~820 FPS (single model, batched)
- Full pipeline (detect + align + embed): ~200-400 RPS

Usage:
    source .venv/bin/activate && python tests/bench_scrfd_pipeline.py
    source .venv/bin/activate && python tests/bench_scrfd_pipeline.py --clients 16 --duration 30
    source .venv/bin/activate && python tests/bench_scrfd_pipeline.py --images 100

Attribution:
    - SCRFD model: InsightFace (https://github.com/deepinsight/insightface), MIT License
    - Reference architecture: hiennguyen9874/triton-face-recognition
    - Benchmark methodology: InsightFace-REST (https://github.com/SthPhoenix/InsightFace-REST)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests


# =============================================================================
# Configuration
# =============================================================================

API_BASE = os.environ.get('API_BASE', 'http://localhost:4603')
TEST_IMAGE_DIR = Path('test_images')
LFW_DIR = TEST_IMAGE_DIR / 'faces' / 'lfw-deepfunneled'


def collect_face_images(max_images: int = 100) -> list[Path]:
    """Collect face images from LFW dataset or test_images."""
    images: list[Path] = []

    # Try LFW first (most realistic)
    if LFW_DIR.exists():
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            images.extend(LFW_DIR.rglob(ext))
            if len(images) >= max_images:
                break

    # Fallback to test_images
    if not images:
        for candidate in [TEST_IMAGE_DIR / 'faces', TEST_IMAGE_DIR]:
            if candidate.exists():
                for ext in ('*.jpg', '*.jpeg', '*.png'):
                    images.extend(candidate.glob(ext))

    return images[:max_images]


def load_image_bytes(image_path: Path) -> bytes:
    """Load image as bytes."""
    with open(image_path, 'rb') as f:
        return f.read()


# =============================================================================
# Benchmark Functions
# =============================================================================


def bench_serial(
    image_data: list[tuple[str, bytes]],
    endpoint: str,
    iterations: int,
) -> dict:
    """Benchmark serial (single-client) throughput."""
    latencies = []
    faces_detected = 0
    errors = 0

    for i in range(iterations):
        name, data = image_data[i % len(image_data)]
        files = {'file': (name, data, 'image/jpeg')}

        start = time.time()
        try:
            resp = requests.post(
                f'{API_BASE}{endpoint}',
                files=files,
                timeout=30,
            )
            elapsed = (time.time() - start) * 1000

            if resp.status_code == 200:
                latencies.append(elapsed)
                result = resp.json()
                faces_detected += len(result.get('faces', []))
            else:
                errors += 1
        except Exception:
            errors += 1

    if not latencies:
        return {'error': 'No successful requests'}

    arr = np.array(latencies)
    return {
        'requests': len(latencies),
        'errors': errors,
        'faces_detected': faces_detected,
        'mean_ms': float(arr.mean()),
        'p50_ms': float(np.percentile(arr, 50)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
        'min_ms': float(arr.min()),
        'max_ms': float(arr.max()),
        'rps': 1000.0 / arr.mean(),
    }


def _single_request(image_data: tuple[str, bytes], endpoint: str) -> tuple[float, int, bool]:
    """Make a single request, return (latency_ms, faces, success)."""
    name, data = image_data
    files = {'file': (name, data, 'image/jpeg')}

    start = time.time()
    try:
        resp = requests.post(
            f'{API_BASE}{endpoint}',
            files=files,
            timeout=30,
        )
        elapsed = (time.time() - start) * 1000

        if resp.status_code == 200:
            result = resp.json()
            faces = len(result.get('faces', []))
            return elapsed, faces, True
        return elapsed, 0, False
    except Exception:
        return 0.0, 0, False


def bench_concurrent(
    image_data: list[tuple[str, bytes]],
    endpoint: str,
    num_clients: int,
    duration_sec: int,
) -> dict:
    """Benchmark concurrent multi-client throughput."""
    latencies = []
    faces_detected = 0
    errors = 0
    start_time = time.time()
    request_idx = 0

    with ThreadPoolExecutor(max_workers=num_clients) as executor:
        futures = {}

        # Submit initial batch
        for _ in range(num_clients * 2):
            data = image_data[request_idx % len(image_data)]
            future = executor.submit(_single_request, data, endpoint)
            futures[future] = request_idx
            request_idx += 1

        # Process completions and submit new requests
        while time.time() - start_time < duration_sec:
            done = [f for f in list(futures.keys()) if f.done()]

            for f in done:
                del futures[f]
                elapsed, faces, success = f.result()
                if success:
                    latencies.append(elapsed)
                    faces_detected += faces
                else:
                    errors += 1

                # Submit replacement
                if time.time() - start_time < duration_sec:
                    data = image_data[request_idx % len(image_data)]
                    new_future = executor.submit(_single_request, data, endpoint)
                    futures[new_future] = request_idx
                    request_idx += 1

            time.sleep(0.001)  # Avoid busy loop

        # Collect remaining
        for f in as_completed(futures.keys(), timeout=10):
            elapsed, faces, success = f.result()
            if success:
                latencies.append(elapsed)
                faces_detected += faces
            else:
                errors += 1

    total_time = time.time() - start_time

    if not latencies:
        return {'error': 'No successful requests'}

    arr = np.array(latencies)
    return {
        'requests': len(latencies),
        'errors': errors,
        'faces_detected': faces_detected,
        'duration_sec': total_time,
        'mean_ms': float(arr.mean()),
        'p50_ms': float(np.percentile(arr, 50)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
        'min_ms': float(arr.min()),
        'max_ms': float(arr.max()),
        'rps': len(latencies) / total_time,
        'clients': num_clients,
    }


def print_results(title: str, results: dict) -> None:
    """Pretty-print benchmark results."""
    print(f'\n  {title}')
    print(f'  {"=" * 50}')

    if 'error' in results:
        print(f'  ERROR: {results["error"]}')
        return

    print(f'  Requests:       {results["requests"]}')
    print(f'  Errors:         {results["errors"]}')
    print(f'  Faces detected: {results["faces_detected"]}')
    if 'duration_sec' in results:
        print(f'  Duration:       {results["duration_sec"]:.1f}s')
    if 'clients' in results:
        print(f'  Clients:        {results["clients"]}')
    print(f'  Mean latency:   {results["mean_ms"]:.1f}ms')
    print(f'  P50 latency:    {results["p50_ms"]:.1f}ms')
    print(f'  P95 latency:    {results["p95_ms"]:.1f}ms')
    print(f'  P99 latency:    {results["p99_ms"]:.1f}ms')
    print(f'  Min latency:    {results["min_ms"]:.1f}ms')
    print(f'  Max latency:    {results["max_ms"]:.1f}ms')
    print(f'  Throughput:     {results["rps"]:.1f} RPS')


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description='SCRFD Face Recognition Pipeline Benchmark')
    parser.add_argument(
        '--images', type=int, default=50, help='Number of test images to use (default: 50)'
    )
    parser.add_argument(
        '--serial-iters', type=int, default=50, help='Iterations for serial benchmark (default: 50)'
    )
    parser.add_argument(
        '--clients', type=int, default=8, help='Number of concurrent clients (default: 8)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=15,
        help='Duration for concurrent benchmark in seconds (default: 15)',
    )
    parser.add_argument(
        '--endpoint',
        type=str,
        default='/v1/faces/recognize',
        help='API endpoint to benchmark (default: /v1/faces/recognize)',
    )
    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations (default: 5)')
    args = parser.parse_args()

    print('=' * 60)
    print('SCRFD Face Recognition Pipeline Benchmark')
    print('=' * 60)
    print(f'API:       {API_BASE}')
    print(f'Endpoint:  {args.endpoint}')

    # Check API health
    try:
        resp = requests.get(f'{API_BASE}/health', timeout=5)
        if resp.status_code == 200:
            health = resp.json()
            print(f'API:       v{health.get("version", "?")} - {health.get("status", "?")}')
        else:
            print(f'API:       HTTP {resp.status_code}')
    except requests.ConnectionError:
        print('ERROR: API not reachable. Start services first.')
        return 1

    # Collect images
    images = collect_face_images(args.images)
    if not images:
        print(f'\nNo face images found in {TEST_IMAGE_DIR}')
        print('Expected: test_images/faces/lfw-deepfunneled/*.jpg')
        return 1

    print(f'Images:    {len(images)} loaded')

    # Load image bytes
    image_data = [(img.name, load_image_bytes(img)) for img in images]
    avg_size = np.mean([len(d) for _, d in image_data]) / 1024
    print(f'Avg size:  {avg_size:.0f} KB')

    # Warmup
    print(f'\nWarming up ({args.warmup} requests)...')
    import contextlib

    for i in range(args.warmup):
        name, data = image_data[i % len(image_data)]
        with contextlib.suppress(Exception):
            requests.post(
                f'{API_BASE}{args.endpoint}',
                files={'file': (name, data, 'image/jpeg')},
                timeout=30,
            )

    # Serial benchmark
    print(f'\nRunning serial benchmark ({args.serial_iters} iterations)...')
    serial_results = bench_serial(image_data, args.endpoint, args.serial_iters)
    print_results('Serial (Single Client)', serial_results)

    # Concurrent benchmark
    for n_clients in [4, args.clients, args.clients * 2]:
        print(f'\nRunning concurrent benchmark ({n_clients} clients, {args.duration}s)...')
        concurrent_results = bench_concurrent(image_data, args.endpoint, n_clients, args.duration)
        print_results(f'Concurrent ({n_clients} clients)', concurrent_results)

    # Summary
    print('\n' + '=' * 60)
    print('Benchmark Summary')
    print('=' * 60)
    if 'rps' in serial_results:
        print(f'  Serial RPS:      {serial_results["rps"]:.1f}')
    if 'rps' in concurrent_results:
        print(f'  Peak RPS:        {concurrent_results["rps"]:.1f}')
    print(f'  Images tested:   {len(images)}')
    print()
    print('Reference targets (InsightFace-REST on RTX 4090):')
    print('  SCRFD-10G detection only:  ~820 FPS')
    print('  Full pipeline (det+align+embed): ~200-400 RPS')

    return 0


if __name__ == '__main__':
    sys.exit(main())
