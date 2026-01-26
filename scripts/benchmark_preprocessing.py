#!/usr/bin/env python3
"""
Benchmark: CPU vs DALI GPU Preprocessing

Compares preprocessing speed between:
1. CPU: cv2 decode + cv2 letterbox resize (Ultralytics-style)
2. GPU DALI: nvJPEG decode + GPU warp_affine (current)

Usage:
    python scripts/benchmark_preprocessing.py --images-dir /mnt/nvm/KILLBOY_SAMPLE_PICTURES --max-images 500
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def find_images(directory: Path, max_images: int = 500) -> list[Path]:
    """Find image files in directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                images.append(Path(root) / f)
                if len(images) >= max_images:
                    return images
    return sorted(images)


def letterbox_cpu(
    image: np.ndarray,
    new_shape: tuple = (640, 640),
    color: tuple = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> tuple[np.ndarray, tuple, tuple]:
    """
    Ultralytics-style letterbox resize on CPU.

    Returns:
        Resized image, ratio (w, h), padding (dw, dh)
    """
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, ratio, (dw, dh)


def preprocess_single_cpu(image_path: Path, yolo_size: int = 640, clip_size: int = 256) -> dict:
    """
    Preprocess single image on CPU for YOLO + CLIP.

    Returns dict with:
    - yolo_tensor: [3, 640, 640] float32 normalized
    - clip_tensor: [3, 256, 256] float32 normalized
    - orig_shape: (h, w)
    """
    # Read and decode
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # YOLO letterbox (640x640)
    yolo_img, _, _ = letterbox_cpu(img_rgb, new_shape=(yolo_size, yolo_size))
    yolo_tensor = yolo_img.transpose(2, 0, 1).astype(np.float32) / 255.0

    # CLIP center crop (256x256)
    # Resize so shorter edge = 256, then center crop
    h, w = img_rgb.shape[:2]
    scale = clip_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    clip_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop
    start_h = (new_h - clip_size) // 2
    start_w = (new_w - clip_size) // 2
    clip_crop = clip_resized[start_h:start_h + clip_size, start_w:start_w + clip_size]
    clip_tensor = clip_crop.transpose(2, 0, 1).astype(np.float32) / 255.0

    return {
        'yolo_tensor': yolo_tensor,
        'clip_tensor': clip_tensor,
        'orig_shape': (orig_h, orig_w),
    }


def benchmark_cpu_preprocessing(images: list[Path], num_workers: int = 16) -> dict:
    """Benchmark CPU preprocessing with ThreadPoolExecutor."""
    print(f"\n=== CPU Preprocessing Benchmark ===")
    print(f"Images: {len(images)}, Workers: {num_workers}")

    start = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(preprocess_single_cpu, images))

    elapsed = time.time() - start
    successful = sum(1 for r in results if r is not None)

    throughput = successful / elapsed

    print(f"Processed: {successful}/{len(images)}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} images/sec")
    print(f"Per image: {1000 * elapsed / successful:.1f} ms")

    return {
        'method': 'cpu',
        'images': len(images),
        'successful': successful,
        'elapsed': elapsed,
        'throughput': throughput,
        'ms_per_image': 1000 * elapsed / successful,
    }


def benchmark_cpu_decode_only(images: list[Path], num_workers: int = 16) -> dict:
    """Benchmark just JPEG decode on CPU."""
    print(f"\n=== CPU JPEG Decode Only ===")
    print(f"Images: {len(images)}, Workers: {num_workers}")

    def decode_only(path: Path) -> bool:
        img = cv2.imread(str(path))
        return img is not None

    start = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(decode_only, images))

    elapsed = time.time() - start
    successful = sum(results)
    throughput = successful / elapsed

    print(f"Decoded: {successful}/{len(images)}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} images/sec")
    print(f"Per image: {1000 * elapsed / successful:.1f} ms")

    return {
        'method': 'cpu_decode_only',
        'images': len(images),
        'successful': successful,
        'elapsed': elapsed,
        'throughput': throughput,
        'ms_per_image': 1000 * elapsed / successful,
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocessing Benchmark')
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--max-images', type=int, default=500)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    images = find_images(Path(args.images_dir), args.max_images)
    print(f"Found {len(images)} images in {args.images_dir}")

    if not images:
        print("No images found!")
        return 1

    # Run benchmarks
    results = []

    # CPU decode only
    results.append(benchmark_cpu_decode_only(images, args.workers))

    # CPU full preprocess (decode + letterbox + normalize)
    results.append(benchmark_cpu_preprocessing(images, args.workers))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'Throughput':>15} {'ms/image':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['method']:<25} {r['throughput']:>12.1f}/sec {r['ms_per_image']:>10.1f}ms")

    return 0


if __name__ == '__main__':
    sys.exit(main())
