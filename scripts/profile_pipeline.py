#!/usr/bin/env python3
"""
Deep profiling of the unified inference pipeline.

Measures precise timings for each step:
1. Image decode
2. CPU preprocessing (YOLO letterbox, CLIP center crop, HD resize)
3. Triton model calls (YOLO, CLIP, unified_embedding_extractor)
4. Response parsing
5. Total end-to-end time

Usage:
    python scripts/profile_pipeline.py --image /path/to/image.jpg
    python scripts/profile_pipeline.py --images-dir /path/to/images --num-images 10
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


def profile_cpu_preprocessing(image_bytes: bytes) -> dict:
    """Profile CPU preprocessing steps individually."""
    timings = {}

    # Step 1: Decode
    start = time.perf_counter()
    img_array = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    timings['decode'] = (time.perf_counter() - start) * 1000

    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Step 2: YOLO letterbox (640x640)
    start = time.perf_counter()
    target_size = 640
    scale = min(target_size / orig_h, target_size / orig_w)
    scale = min(scale, 1.0)  # Don't upscale
    new_w, new_h = round(orig_w * scale), round(orig_h * scale)

    if scale != 1.0:
        yolo_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        yolo_resized = img_rgb

    pad_w = (target_size - new_w) / 2.0
    pad_h = (target_size - new_h) / 2.0
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)

    yolo_letterboxed = cv2.copyMakeBorder(yolo_resized, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=(114, 114, 114))
    yolo_tensor = yolo_letterboxed.astype(np.float32) / 255.0
    yolo_tensor = np.transpose(yolo_tensor, (2, 0, 1))
    timings['yolo_letterbox'] = (time.perf_counter() - start) * 1000

    # Step 3: CLIP center crop (256x256)
    start = time.perf_counter()
    clip_size = 256
    clip_scale = clip_size / min(orig_h, orig_w)
    clip_new_w, clip_new_h = int(orig_w * clip_scale), int(orig_h * clip_scale)
    clip_resized = cv2.resize(img_rgb, (clip_new_w, clip_new_h), interpolation=cv2.INTER_LINEAR)

    start_x = (clip_new_w - clip_size) // 2
    start_y = (clip_new_h - clip_size) // 2
    clip_cropped = clip_resized[start_y:start_y + clip_size, start_x:start_x + clip_size]
    clip_tensor = clip_cropped.astype(np.float32) / 255.0
    clip_tensor = np.transpose(clip_tensor, (2, 0, 1))
    timings['clip_center_crop'] = (time.perf_counter() - start) * 1000

    # Step 4: HD resize (max 1920px)
    start = time.perf_counter()
    max_size = 1920
    hd_scale = min(max_size / max(orig_h, orig_w), 1.0)
    if hd_scale < 1.0:
        hd_new_w, hd_new_h = round(orig_w * hd_scale), round(orig_h * hd_scale)
        hd_resized = cv2.resize(img_rgb, (hd_new_w, hd_new_h), interpolation=cv2.INTER_LINEAR)
    else:
        hd_resized = img_rgb
    hd_tensor = hd_resized.astype(np.float32) / 255.0
    hd_tensor = np.transpose(hd_tensor, (2, 0, 1))
    timings['hd_resize'] = (time.perf_counter() - start) * 1000

    # Step 5: Affine matrix calculation
    start = time.perf_counter()
    affine_matrix = np.array([[scale, 0, left], [0, scale, top]], dtype=np.float32)
    timings['affine_calc'] = (time.perf_counter() - start) * 1000

    timings['total_preprocess'] = sum([
        timings['decode'], timings['yolo_letterbox'],
        timings['clip_center_crop'], timings['hd_resize'], timings['affine_calc']
    ])

    return {
        'timings': timings,
        'yolo_tensor': yolo_tensor,
        'clip_tensor': clip_tensor,
        'hd_tensor': hd_tensor,
        'affine_matrix': affine_matrix,
        'orig_shape': (orig_h, orig_w),
        'scale': scale,
        'padding': (pad_w, pad_h),
    }


def profile_triton_calls(client, prep_result: dict) -> dict:
    """Profile individual Triton model calls."""
    from tritonclient.grpc import InferInput, InferRequestedOutput

    timings = {}

    yolo_tensor = prep_result['yolo_tensor']
    clip_tensor = prep_result['clip_tensor']
    hd_tensor = prep_result['hd_tensor']
    affine_matrix = prep_result['affine_matrix']

    # Step 1: YOLO detection
    start = time.perf_counter()
    yolo_input = InferInput('images', [1, 3, 640, 640], 'FP32')
    yolo_input.set_data_from_numpy(yolo_tensor[np.newaxis].astype(np.float32))

    yolo_outputs = [
        InferRequestedOutput('num_dets'),
        InferRequestedOutput('det_boxes'),
        InferRequestedOutput('det_scores'),
        InferRequestedOutput('det_classes'),
    ]

    yolo_response = client._client.infer('yolov11_small_trt_end2end', [yolo_input], outputs=yolo_outputs)
    num_dets = int(yolo_response.as_numpy('num_dets').flatten()[0])
    det_boxes = yolo_response.as_numpy('det_boxes')
    det_scores = yolo_response.as_numpy('det_scores')
    det_classes = yolo_response.as_numpy('det_classes')
    timings['yolo_detection'] = (time.perf_counter() - start) * 1000

    # Step 2: MobileCLIP global embedding
    start = time.perf_counter()
    clip_input = InferInput('images', [1, 3, 256, 256], 'FP32')
    clip_input.set_data_from_numpy(clip_tensor[np.newaxis].astype(np.float32))

    clip_outputs = [InferRequestedOutput('image_embeddings')]
    clip_response = client._client.infer('mobileclip2_s2_image_encoder', [clip_input], outputs=clip_outputs)
    global_embedding = clip_response.as_numpy('image_embeddings')
    timings['clip_global'] = (time.perf_counter() - start) * 1000

    # Step 3: unified_embedding_extractor (per-box embeddings + face detection)
    start = time.perf_counter()

    # Prepare inputs for unified_embedding_extractor
    orig_input = InferInput('original_image', [1] + list(hd_tensor.shape), 'FP32')
    orig_input.set_data_from_numpy(hd_tensor[np.newaxis].astype(np.float32))

    boxes_input = InferInput('det_boxes', list(det_boxes.shape), 'FP32')
    boxes_input.set_data_from_numpy(det_boxes.astype(np.float32))

    classes_input = InferInput('det_classes', list(det_classes.shape), 'INT32')
    classes_input.set_data_from_numpy(det_classes.astype(np.int32))

    scores_input = InferInput('det_scores', list(det_scores.shape), 'FP32')
    scores_input.set_data_from_numpy(det_scores.astype(np.float32))

    num_dets_input = InferInput('num_dets', [1, 1], 'INT32')
    num_dets_input.set_data_from_numpy(np.array([[num_dets]], dtype=np.int32))

    affine_input = InferInput('affine_matrix', [1, 2, 3], 'FP32')
    affine_input.set_data_from_numpy(affine_matrix[np.newaxis].astype(np.float32))

    extractor_outputs = [
        InferRequestedOutput('box_embeddings'),
        InferRequestedOutput('normalized_boxes'),
        InferRequestedOutput('num_faces'),
        InferRequestedOutput('face_embeddings'),
        InferRequestedOutput('face_boxes'),
        InferRequestedOutput('face_landmarks'),
        InferRequestedOutput('face_scores'),
        InferRequestedOutput('face_person_idx'),
    ]

    extractor_response = client._client.infer(
        'unified_embedding_extractor',
        [orig_input, boxes_input, classes_input, scores_input, num_dets_input, affine_input],
        outputs=extractor_outputs
    )

    box_embeddings = extractor_response.as_numpy('box_embeddings')
    num_faces = int(extractor_response.as_numpy('num_faces').flatten()[0])
    timings['unified_embedding_extractor'] = (time.perf_counter() - start) * 1000

    timings['total_triton'] = timings['yolo_detection'] + timings['clip_global'] + timings['unified_embedding_extractor']

    return {
        'timings': timings,
        'num_dets': num_dets,
        'num_faces': num_faces,
    }


def profile_dali_pipeline(client, image_bytes: bytes) -> dict:
    """Profile the DALI-based unified pipeline for comparison."""
    timings = {}

    start = time.perf_counter()
    result = client.infer_unified_complete(image_bytes)
    timings['total_dali_pipeline'] = (time.perf_counter() - start) * 1000

    return {
        'timings': timings,
        'num_dets': result['num_dets'],
        'num_faces': result['num_faces'],
    }


def print_timings_chart(cpu_timings: dict, triton_timings: dict, dali_timings: dict):
    """Print a visual chart of timings."""
    print("\n" + "=" * 70)
    print("PIPELINE TIMING BREAKDOWN")
    print("=" * 70)

    print("\n📊 CPU PREPROCESSING (client-side)")
    print("-" * 50)
    for key, value in cpu_timings.items():
        bar_len = int(value / 2)  # Scale for display
        bar = "█" * bar_len
        print(f"  {key:25s} {value:8.2f} ms  {bar}")

    print("\n📊 TRITON INFERENCE (server-side)")
    print("-" * 50)
    for key, value in triton_timings.items():
        bar_len = int(value / 5)  # Scale for display
        bar = "█" * bar_len
        print(f"  {key:25s} {value:8.2f} ms  {bar}")

    print("\n📊 DALI PIPELINE (all-in-one)")
    print("-" * 50)
    for key, value in dali_timings.items():
        bar_len = int(value / 5)  # Scale for display
        bar = "█" * bar_len
        print(f"  {key:25s} {value:8.2f} ms  {bar}")

    # Summary comparison
    cpu_total = cpu_timings.get('total_preprocess', 0)
    triton_total = triton_timings.get('total_triton', 0)
    dali_total = dali_timings.get('total_dali_pipeline', 0)
    cpu_plus_triton = cpu_total + triton_total

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"  CPU Preprocess + Triton:  {cpu_plus_triton:8.2f} ms")
    print(f"    - CPU preprocessing:    {cpu_total:8.2f} ms")
    print(f"    - Triton inference:     {triton_total:8.2f} ms")
    print(f"  DALI Pipeline (baseline): {dali_total:8.2f} ms")
    print(f"  Difference:               {cpu_plus_triton - dali_total:+8.2f} ms")

    if cpu_plus_triton < dali_total:
        speedup = dali_total / cpu_plus_triton
        print(f"  CPU+Triton is {speedup:.2f}x FASTER")
    else:
        slowdown = cpu_plus_triton / dali_total
        print(f"  CPU+Triton is {slowdown:.2f}x SLOWER")

    # Bottleneck analysis
    print("\n" + "=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)

    all_steps = {
        'CPU decode': cpu_timings.get('decode', 0),
        'CPU YOLO letterbox': cpu_timings.get('yolo_letterbox', 0),
        'CPU CLIP center crop': cpu_timings.get('clip_center_crop', 0),
        'CPU HD resize': cpu_timings.get('hd_resize', 0),
        'Triton YOLO detection': triton_timings.get('yolo_detection', 0),
        'Triton CLIP global': triton_timings.get('clip_global', 0),
        'Triton unified_extractor': triton_timings.get('unified_embedding_extractor', 0),
    }

    sorted_steps = sorted(all_steps.items(), key=lambda x: x[1], reverse=True)

    print("  Top bottlenecks:")
    for i, (step, time_ms) in enumerate(sorted_steps[:5], 1):
        pct = (time_ms / cpu_plus_triton) * 100 if cpu_plus_triton > 0 else 0
        print(f"    {i}. {step:30s} {time_ms:8.2f} ms ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Profile unified inference pipeline')
    parser.add_argument('--image', type=str, help='Single image to profile')
    parser.add_argument('--images-dir', type=str, help='Directory of images')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images to profile')
    parser.add_argument('--triton-url', type=str, default='localhost:4601', help='Triton gRPC URL')
    args = parser.parse_args()

    # Import Triton client
    from src.clients.triton_client import get_triton_client
    client = get_triton_client(args.triton_url)

    # Get image(s) to profile
    if args.image:
        image_paths = [args.image]
    elif args.images_dir:
        image_paths = sorted([
            os.path.join(args.images_dir, f)
            for f in os.listdir(args.images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])[:args.num_images]
    else:
        print("Please provide --image or --images-dir")
        return 1

    print(f"Profiling {len(image_paths)} image(s)...")

    # Aggregate timings
    all_cpu_timings = []
    all_triton_timings = []
    all_dali_timings = []

    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Profiling: {os.path.basename(image_path)}")

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Profile CPU preprocessing
        cpu_result = profile_cpu_preprocessing(image_bytes)
        all_cpu_timings.append(cpu_result['timings'])

        # Profile Triton calls with preprocessed data
        triton_result = profile_triton_calls(client, cpu_result)
        all_triton_timings.append(triton_result['timings'])

        # Profile DALI pipeline for comparison
        dali_result = profile_dali_pipeline(client, image_bytes)
        all_dali_timings.append(dali_result['timings'])

        print(f"  CPU prep: {cpu_result['timings']['total_preprocess']:.1f}ms, "
              f"Triton: {triton_result['timings']['total_triton']:.1f}ms, "
              f"DALI: {dali_result['timings']['total_dali_pipeline']:.1f}ms")
        print(f"  Detections: {triton_result['num_dets']}, Faces: {triton_result['num_faces']}")

    # Calculate averages
    avg_cpu = {k: np.mean([t[k] for t in all_cpu_timings]) for k in all_cpu_timings[0]}
    avg_triton = {k: np.mean([t[k] for t in all_triton_timings]) for k in all_triton_timings[0]}
    avg_dali = {k: np.mean([t[k] for t in all_dali_timings]) for k in all_dali_timings[0]}

    # Print detailed chart
    print_timings_chart(avg_cpu, avg_triton, avg_dali)

    return 0


if __name__ == '__main__':
    sys.exit(main())
