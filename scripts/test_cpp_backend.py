#!/usr/bin/env python3
"""Test the C++ face pipeline backend directly via gRPC."""

import sys
import time
import numpy as np
import cv2
from PIL import Image
import tritonclient.grpc as grpcclient

# Triton server address (inside container, use triton-api:8001)
# From host: localhost:4601
# From yolo-api container: triton-api:8001
import os
TRITON_URL = os.environ.get("TRITON_URL", "triton-api:8001")

def letterbox_image(img, target_size=640):
    """Letterbox resize to target_size x target_size with padding."""
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    return padded, scale, pad_x, pad_y


def preprocess_for_yolo(img):
    """Preprocess image for YOLO detection."""
    # BGR to RGB, HWC to CHW, normalize to [0,1]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_chw, axis=0)  # Add batch dimension


def preprocess_hd_image(img):
    """Preprocess HD image for cropping - normalize to [0,1]."""
    # BGR to RGB, HWC to CHW, normalize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_chw, axis=0)  # [1, 3, H, W]


def test_cpp_backend(image_path: str):
    """Test the C++ face pipeline backend."""
    print(f"\n{'='*60}")
    print("Testing C++ Face Pipeline Backend")
    print(f"{'='*60}")

    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"ERROR: Could not load image: {image_path}")
        return False

    orig_h, orig_w = img_bgr.shape[:2]
    print(f"Image: {image_path}")
    print(f"Original size: {orig_w}x{orig_h}")

    # Prepare inputs
    # 1. Letterbox for YOLO detection
    letterbox_img, scale, pad_x, pad_y = letterbox_image(img_bgr, 640)
    face_images = preprocess_for_yolo(letterbox_img)
    print(f"Face images (letterbox): {face_images.shape}")

    # 2. HD image for cropping
    original_image = preprocess_hd_image(img_bgr)
    print(f"Original image (HD): {original_image.shape}")

    # 3. Original shape
    orig_shape = np.array([orig_h, orig_w], dtype=np.int32)
    print(f"Original shape: {orig_shape}")

    # Create Triton client
    try:
        client = grpcclient.InferenceServerClient(url=TRITON_URL)
    except Exception as e:
        print(f"ERROR: Could not connect to Triton: {e}")
        return False

    # Prepare inference inputs
    inputs = [
        grpcclient.InferInput("face_images", face_images.shape, "FP32"),
        grpcclient.InferInput("original_image", original_image.shape, "FP32"),
        grpcclient.InferInput("orig_shape", orig_shape.shape, "INT32"),
    ]
    inputs[0].set_data_from_numpy(face_images)
    inputs[1].set_data_from_numpy(original_image)
    inputs[2].set_data_from_numpy(orig_shape)

    # Request outputs
    outputs = [
        grpcclient.InferRequestedOutput("num_faces"),
        grpcclient.InferRequestedOutput("boxes"),
        grpcclient.InferRequestedOutput("landmarks"),
        grpcclient.InferRequestedOutput("scores"),
        grpcclient.InferRequestedOutput("embeddings"),
        grpcclient.InferRequestedOutput("quality"),
    ]

    # Run inference
    print("\nRunning inference...")
    start = time.perf_counter()
    try:
        result = client.infer(model_name="face_pipeline_cpp", inputs=inputs, outputs=outputs)
        elapsed = (time.perf_counter() - start) * 1000
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get outputs
    num_faces = result.as_numpy("num_faces")[0]
    boxes = result.as_numpy("boxes")
    landmarks = result.as_numpy("landmarks")
    scores = result.as_numpy("scores")
    embeddings = result.as_numpy("embeddings")
    quality = result.as_numpy("quality")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Inference time: {elapsed:.2f} ms")
    print(f"Faces detected: {num_faces}")

    if num_faces > 0:
        print(f"\nBoxes shape: {boxes.shape}")
        print(f"Landmarks shape: {landmarks.shape}")
        print(f"Scores shape: {scores.shape}")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Quality shape: {quality.shape}")

        for i in range(min(num_faces, 5)):  # Show first 5 faces
            print(f"\nFace {i+1}:")
            print(f"  Box (HD coords): [{boxes[i, 0]:.1f}, {boxes[i, 1]:.1f}, {boxes[i, 2]:.1f}, {boxes[i, 3]:.1f}]")
            print(f"  Score: {scores[i]:.4f}")
            print(f"  Quality: {quality[i]:.4f}")
            print(f"  Embedding norm: {np.linalg.norm(embeddings[i]):.4f} (should be ~1.0)")

    return True


def benchmark_cpp_backend(image_path: str, num_iterations: int = 100):
    """Benchmark the C++ face pipeline backend."""
    print(f"\n{'='*60}")
    print("Benchmarking C++ Face Pipeline Backend")
    print(f"{'='*60}")

    # Load and preprocess image
    img_bgr = cv2.imread(image_path)
    orig_h, orig_w = img_bgr.shape[:2]
    letterbox_img, _, _, _ = letterbox_image(img_bgr, 640)
    face_images = preprocess_for_yolo(letterbox_img)
    original_image = preprocess_hd_image(img_bgr)
    orig_shape = np.array([orig_h, orig_w], dtype=np.int32)

    # Create client
    client = grpcclient.InferenceServerClient(url=TRITON_URL)

    # Prepare inputs
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

    # Warmup
    print("Warming up...")
    for _ in range(10):
        client.infer(model_name="face_pipeline_cpp", inputs=inputs, outputs=outputs)

    # Benchmark
    print(f"Running {num_iterations} iterations...")
    latencies = []
    start_total = time.perf_counter()

    for i in range(num_iterations):
        start = time.perf_counter()
        result = client.infer(model_name="face_pipeline_cpp", inputs=inputs, outputs=outputs)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{num_iterations} done...")

    total_time = time.perf_counter() - start_total

    # Statistics
    latencies = np.array(latencies)
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} s")
    print(f"Throughput: {num_iterations / total_time:.1f} RPS")
    print(f"Latency (ms):")
    print(f"  Mean:   {np.mean(latencies):.2f}")
    print(f"  Median: {np.median(latencies):.2f}")
    print(f"  P95:    {np.percentile(latencies, 95):.2f}")
    print(f"  P99:    {np.percentile(latencies, 99):.2f}")
    print(f"  Min:    {np.min(latencies):.2f}")
    print(f"  Max:    {np.max(latencies):.2f}")


if __name__ == "__main__":
    # Default test image
    test_image = "/mnt/nvm/repos/triton-api/test_images/benchmark_all/0_Parade_marchingband_1_1018.jpg"

    if len(sys.argv) > 1:
        test_image = sys.argv[1]

    # Test functionality first
    if test_cpp_backend(test_image):
        print("\n" + "="*60)
        print("FUNCTIONAL TEST PASSED")
        print("="*60)

        # Run benchmark
        benchmark_cpp_backend(test_image, num_iterations=100)
    else:
        print("\n" + "="*60)
        print("FUNCTIONAL TEST FAILED")
        print("="*60)
        sys.exit(1)
