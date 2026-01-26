#!/usr/bin/env python3
"""Concurrent benchmark for C++ face pipeline backend."""

import sys
import time
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import tritonclient.grpc as grpcclient
import os

TRITON_URL = os.environ.get("TRITON_URL", "triton-api:8001")


def letterbox_image(img, target_size=640):
    """Letterbox resize to target_size x target_size with padding."""
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return padded


def preprocess_for_yolo(img):
    """Preprocess image for YOLO detection."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_chw, axis=0)


def preprocess_hd_image(img):
    """Preprocess HD image for cropping."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_chw, axis=0)


def run_inference(client, face_images, original_image, orig_shape):
    """Run single inference."""
    inputs = [
        grpcclient.InferInput("face_images", face_images.shape, "FP32"),
        grpcclient.InferInput("original_image", original_image.shape, "FP32"),
        grpcclient.InferInput("orig_shape", orig_shape.shape, "INT32"),
    ]
    inputs[0].set_data_from_numpy(face_images)
    inputs[1].set_data_from_numpy(original_image)
    inputs[2].set_data_from_numpy(orig_shape)

    outputs = [
        grpcclient.InferRequestedOutput("num_faces"),
        grpcclient.InferRequestedOutput("embeddings"),
    ]

    result = client.infer(model_name="face_pipeline_cpp", inputs=inputs, outputs=outputs)
    return result.as_numpy("num_faces")[0]


def benchmark_concurrent(image_path: str, num_requests: int = 100, concurrency: int = 16):
    """Benchmark with concurrent requests."""
    print(f"\n{'='*60}")
    print(f"Benchmarking C++ Face Pipeline Backend")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Requests: {num_requests}")
    print(f"Concurrency: {concurrency}")

    # Load and preprocess image
    img_bgr = cv2.imread(image_path)
    orig_h, orig_w = img_bgr.shape[:2]
    letterbox_img = letterbox_image(img_bgr, 640)
    face_images = preprocess_for_yolo(letterbox_img)
    original_image = preprocess_hd_image(img_bgr)
    orig_shape = np.array([orig_h, orig_w], dtype=np.int32)

    # Create a client pool
    clients = [grpcclient.InferenceServerClient(url=TRITON_URL) for _ in range(concurrency)]

    # Warmup
    print("Warming up...")
    for i in range(min(10, concurrency)):
        run_inference(clients[i], face_images, original_image, orig_shape)

    # Benchmark
    print("Running benchmark...")
    total_faces = 0
    latencies = []

    def run_task(idx):
        client = clients[idx % concurrency]
        start = time.perf_counter()
        num_faces = run_inference(client, face_images, original_image, orig_shape)
        elapsed = (time.perf_counter() - start) * 1000
        return num_faces, elapsed

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(run_task, range(num_requests)))
    total_time = time.perf_counter() - start_total

    for num_faces, latency in results:
        total_faces += num_faces
        latencies.append(latency)

    latencies = np.array(latencies)
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {num_requests/total_time:.1f} RPS")
    print(f"Total faces detected: {total_faces}")
    print(f"Latency (ms):")
    print(f"  Mean:   {np.mean(latencies):.2f}")
    print(f"  Median: {np.median(latencies):.2f}")
    print(f"  P95:    {np.percentile(latencies, 95):.2f}")
    print(f"  P99:    {np.percentile(latencies, 99):.2f}")
    print(f"  Min:    {np.min(latencies):.2f}")
    print(f"  Max:    {np.max(latencies):.2f}")


if __name__ == "__main__":
    image_path = "/app/test_images/zidane.jpg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    num_requests = 200
    if len(sys.argv) > 2:
        num_requests = int(sys.argv[2])

    concurrency = 16
    if len(sys.argv) > 3:
        concurrency = int(sys.argv[3])

    benchmark_concurrent(image_path, num_requests, concurrency)
