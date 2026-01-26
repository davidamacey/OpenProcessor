#!/usr/bin/env python3
"""
Simple benchmark script for Triton API endpoints.

Usage:
    python benchmark.py [--endpoint /detect] [--requests 100] [--concurrent 10]
"""

import argparse
import asyncio
import time
from pathlib import Path
from statistics import mean, stdev

import httpx


async def benchmark_endpoint(
    url: str,
    image_path: Path,
    num_requests: int,
    concurrency: int,
) -> dict:
    """Benchmark a single endpoint."""
    latencies = []
    errors = 0

    async def make_request(client: httpx.AsyncClient, semaphore: asyncio.Semaphore):
        nonlocal errors
        async with semaphore:
            try:
                with open(image_path, "rb") as f:
                    files = {"image": ("test.jpg", f, "image/jpeg")}
                    start = time.perf_counter()
                    response = await client.post(url, files=files)
                    elapsed = (time.perf_counter() - start) * 1000  # ms

                    if response.status_code == 200:
                        latencies.append(elapsed)
                    else:
                        errors += 1
            except Exception:
                errors += 1

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=30.0) as client:
        start_time = time.perf_counter()
        tasks = [make_request(client, semaphore) for _ in range(num_requests)]
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

    if latencies:
        sorted_latencies = sorted(latencies)
        p50_idx = int(len(sorted_latencies) * 0.50)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)

        return {
            "requests": num_requests,
            "successful": len(latencies),
            "errors": errors,
            "total_time_s": total_time,
            "rps": len(latencies) / total_time,
            "latency_mean_ms": mean(latencies),
            "latency_stdev_ms": stdev(latencies) if len(latencies) > 1 else 0,
            "latency_p50_ms": sorted_latencies[p50_idx],
            "latency_p95_ms": sorted_latencies[p95_idx],
            "latency_p99_ms": sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)],
        }

    return {"requests": num_requests, "successful": 0, "errors": errors}


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton API endpoints")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=4603, help="API port")
    parser.add_argument("--endpoint", default="/detect", help="Endpoint to benchmark")
    parser.add_argument("--image", default="/app/test_images/sample.jpg", help="Test image path")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}{args.endpoint}"
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Benchmarking: {url}")
    print(f"Image: {image_path}")
    print(f"Requests: {args.requests}, Concurrency: {args.concurrent}")
    print("-" * 60)

    results = await benchmark_endpoint(url, image_path, args.requests, args.concurrent)

    print(f"Successful requests: {results['successful']}/{results['requests']}")
    print(f"Errors: {results['errors']}")

    if results['successful'] > 0:
        print(f"Total time: {results['total_time_s']:.2f}s")
        print(f"Throughput: {results['rps']:.2f} RPS")
        print(f"Latency (mean): {results['latency_mean_ms']:.2f}ms")
        print(f"Latency (stdev): {results['latency_stdev_ms']:.2f}ms")
        print(f"Latency (p50): {results['latency_p50_ms']:.2f}ms")
        print(f"Latency (p95): {results['latency_p95_ms']:.2f}ms")
        print(f"Latency (p99): {results['latency_p99_ms']:.2f}ms")


if __name__ == "__main__":
    asyncio.run(main())
