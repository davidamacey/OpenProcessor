#!/usr/bin/env python3
"""
Benchmark script to compare face recognition endpoints.

Compares:
1. Fast Face (direct gRPC) - /track_e/faces/fast/recognize
2. Python BLS - /track_e/faces/recognize
3. YOLO11 Face Pipeline - /track_e/faces/yolo11/recognize
"""

import argparse
import os
import time
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


API_BASE = os.environ.get("API_BASE", "http://localhost:4603")


def benchmark_endpoint(
    endpoint: str,
    image_path: str,
    num_requests: int,
    concurrency: int,
    file_param: str = "image"
) -> dict:
    """Benchmark a single endpoint."""

    # Read image once
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    latencies = []
    successful = 0
    total_faces = 0
    errors = []

    def make_request(idx: int) -> tuple:
        """Make a single request."""
        start = time.perf_counter()
        try:
            files = {file_param: ("image.jpg", image_bytes, "image/jpeg")}
            response = requests.post(
                f"{API_BASE}{endpoint}",
                files=files,
                params={"confidence": 0.5},
                timeout=30
            )
            elapsed = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                # Handle different response formats
                num_faces = data.get("num_faces", 0)
                if num_faces == 0 and "faces" in data:
                    num_faces = len(data["faces"])
                return True, elapsed, num_faces, None
            else:
                return False, elapsed, 0, f"HTTP {response.status_code}"
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return False, elapsed, 0, str(e)

    # Warmup
    print(f"  Warming up...", end="", flush=True)
    for _ in range(min(5, concurrency)):
        make_request(0)
    print(" done")

    # Benchmark
    print(f"  Running {num_requests} requests with {concurrency} concurrent...", end="", flush=True)
    start_total = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        for future in as_completed(futures):
            success, latency, faces, error = future.result()
            if success:
                successful += 1
                latencies.append(latency)
                total_faces += faces
            else:
                if error and len(errors) < 5:
                    errors.append(error)

    total_time = time.perf_counter() - start_total
    print(" done")

    latencies = np.array(latencies) if latencies else np.array([0])

    return {
        "endpoint": endpoint,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "successful": successful,
        "total_time": total_time,
        "rps": successful / total_time if total_time > 0 else 0,
        "total_faces": total_faces,
        "latency_mean": float(np.mean(latencies)),
        "latency_median": float(np.median(latencies)),
        "latency_p95": float(np.percentile(latencies, 95)),
        "latency_p99": float(np.percentile(latencies, 99)),
        "latency_min": float(np.min(latencies)),
        "latency_max": float(np.max(latencies)),
        "errors": errors
    }


def print_results(results: dict):
    """Print benchmark results."""
    print(f"\n  Throughput: {results['rps']:.1f} RPS")
    print(f"  Successful: {results['successful']}/{results['num_requests']}")
    print(f"  Total faces: {results['total_faces']}")
    print(f"  Latency (ms):")
    print(f"    Mean:   {results['latency_mean']:.1f}")
    print(f"    Median: {results['latency_median']:.1f}")
    print(f"    P95:    {results['latency_p95']:.1f}")
    print(f"    P99:    {results['latency_p99']:.1f}")
    if results['errors']:
        print(f"  Errors: {results['errors'][:3]}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark face recognition endpoints")
    parser.add_argument("--image", type=str,
                        default="/mnt/nvm/repos/triton-api/test_images/benchmark_all/0_Parade_marchingband_1_1018.jpg",
                        help="Test image path")
    parser.add_argument("--requests", type=int, default=200,
                        help="Number of requests per endpoint")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of concurrent requests")
    args = parser.parse_args()

    if not Path(args.image).exists():
        # Try alternate path
        alt_path = "/app/test_images/benchmark_all/0_Parade_marchingband_1_1018.jpg"
        if Path(alt_path).exists():
            args.image = alt_path
        else:
            print(f"Error: Image not found at {args.image}")
            return 1

    print("=" * 70)
    print("FACE RECOGNITION ENDPOINT BENCHMARK")
    print("=" * 70)
    print(f"Image: {args.image}")
    print(f"Requests: {args.requests}")
    print(f"Concurrency: {args.concurrency}")

    endpoints = [
        ("/track_e/faces/fast/recognize", "file", "Fast Face (direct gRPC, no BLS)"),
        ("/track_e/faces/recognize", "image", "Python BLS Face Pipeline"),
        ("/track_e/faces/yolo11/recognize", "file", "YOLO11 Face Pipeline"),
    ]

    all_results = []

    for endpoint, file_param, name in endpoints:
        print(f"\n{'='*70}")
        print(f"[{len(all_results)+1}] {name}")
        print(f"    Endpoint: {endpoint}")
        print("=" * 70)

        results = benchmark_endpoint(
            endpoint=endpoint,
            image_path=args.image,
            num_requests=args.requests,
            concurrency=args.concurrency,
            file_param=file_param
        )
        results["name"] = name
        all_results.append(results)
        print_results(results)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Endpoint':<40} {'RPS':>10} {'Latency':>12} {'Success':>10}")
    print("-" * 70)

    for r in all_results:
        name = r["name"][:38]
        rps = f"{r['rps']:.1f}"
        latency = f"{r['latency_mean']:.1f}ms"
        success = f"{r['successful']}/{r['num_requests']}"
        print(f"{name:<40} {rps:>10} {latency:>12} {success:>10}")

    # Find best
    if all_results:
        best = max(all_results, key=lambda x: x['rps'])
        print(f"\nBest: {best['name']} at {best['rps']:.1f} RPS")

    return 0


if __name__ == "__main__":
    exit(main())
