#!/usr/bin/env python3
"""
SCRFD Pipeline Integration Test

Tests the full face recognition pipeline:
1. SCRFD face detection with 5-point landmarks
2. Umeyama alignment for ArcFace
3. ArcFace embedding extraction
4. End-to-end API test via /faces/recognize

Run:
    source .venv/bin/activate && python tests/test_scrfd_pipeline.py

Prerequisites:
    - Triton server running with scrfd_10g_bnkps and arcface_w600k_r50 loaded
    - yolo-api running on port 4603
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import requests


# =============================================================================
# Configuration
# =============================================================================

API_BASE = os.environ.get('API_BASE', 'http://localhost:4603')
TRITON_HTTP = os.environ.get('TRITON_HTTP', 'http://localhost:4600')
TRITON_GRPC = os.environ.get('TRITON_GRPC', 'localhost:4601')
TEST_IMAGE_DIR = Path('test_images')

# Thresholds
MIN_CONFIDENCE = 0.3
MIN_EMBEDDING_DIM = 512
LANDMARK_COUNT = 5

# =============================================================================
# Test Helpers
# =============================================================================


def find_test_image() -> Path | None:
    """Find a suitable test image with faces."""
    # Try common test image locations
    candidates = [
        TEST_IMAGE_DIR / 'faces' / 'lfw-deepfunneled',
        TEST_IMAGE_DIR / 'faces',
        TEST_IMAGE_DIR,
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            images = list(candidate.rglob(ext))
            if images:
                return images[0]

    return None


def test_result(name: str, passed: bool, details: str = '') -> bool:
    """Print test result."""
    status = 'PASS' if passed else 'FAIL'
    msg = f'[{status}] {name}'
    if details:
        msg += f' - {details}'
    print(msg)
    return passed


# =============================================================================
# Tests
# =============================================================================


def test_triton_model_ready() -> bool:
    """Check if SCRFD model is loaded on Triton."""
    print('\n--- Triton Model Status ---')

    models = ['scrfd_10g_bnkps', 'arcface_w600k_r50']
    all_ready = True

    for model in models:
        try:
            resp = requests.get(f'{TRITON_HTTP}/v2/models/{model}/ready', timeout=5)
            ready = resp.status_code == 200
            test_result(f'{model} ready', ready, f'HTTP {resp.status_code}')
            if not ready:
                all_ready = False
        except requests.ConnectionError:
            test_result(f'{model} ready', False, 'Triton not reachable')
            all_ready = False

    return all_ready


def test_scrfd_decode_unit() -> bool:
    """Unit test for SCRFD decode logic with synthetic data."""
    print('\n--- SCRFD Decode Unit Test ---')

    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from utils.scrfd_decode import decode_scrfd_outputs

    # Create synthetic output mimicking SCRFD for one face at center
    # Stride 8: 80x80 grid, 2 anchors per cell = 12800
    # Stride 16: 40x40 grid, 2 anchors per cell = 3200
    # Stride 32: 20x20 grid, 2 anchors per cell = 800

    net_outs = {}
    for stride, n_anchors in [(8, 12800), (16, 3200), (32, 800)]:
        # Low confidence by default
        scores = np.full((n_anchors, 1), 0.01, dtype=np.float32)
        bboxes = np.zeros((n_anchors, 4), dtype=np.float32)
        kpss = np.zeros((n_anchors, 10), dtype=np.float32)

        # Place a high-confidence face at center of the feature map
        center_idx = n_anchors // 2
        scores[center_idx] = 0.95

        # Box: 50px on each side (before stride scaling)
        bboxes[center_idx] = [50.0 / stride, 50.0 / stride, 50.0 / stride, 50.0 / stride]

        # Landmarks: simple offsets from anchor center
        kpss[center_idx] = [
            -20 / stride,
            -15 / stride,  # left eye
            20 / stride,
            -15 / stride,  # right eye
            0 / stride,
            5 / stride,  # nose
            -15 / stride,
            25 / stride,  # left mouth
            15 / stride,
            25 / stride,  # right mouth
        ]

        net_outs[f'score_{stride}'] = scores
        net_outs[f'bbox_{stride}'] = bboxes
        net_outs[f'kps_{stride}'] = kpss

    # Decode
    det_scale = 1.0  # No scaling
    boxes, scores, landmarks = decode_scrfd_outputs(
        net_outs, det_scale, det_thresh=0.5, nms_thresh=0.4
    )

    # Validate
    passed = True

    p = test_result('Detections found', len(boxes) > 0, f'{len(boxes)} faces detected')
    passed = passed and p

    if len(boxes) > 0:
        p = test_result('Box shape', boxes.shape[1] == 4, f'Expected [N, 4], got {boxes.shape}')
        passed = passed and p

        p = test_result(
            'Landmark shape',
            landmarks.shape[1:] == (5, 2),
            f'Expected [N, 5, 2], got {landmarks.shape}',
        )
        passed = passed and p

        p = test_result('Score above threshold', float(scores[0]) >= 0.5, f'Score: {scores[0]:.3f}')
        passed = passed and p

        # Check landmarks are non-zero
        lmk_nonzero = np.any(landmarks[0] != 0)
        p = test_result('Landmarks non-zero', lmk_nonzero, f'First landmark: {landmarks[0][0]}')
        passed = passed and p

    return passed


def test_face_alignment_unit() -> bool:
    """Unit test for Umeyama face alignment."""
    print('\n--- Face Alignment Unit Test ---')

    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from utils.face_align import align_face

    passed = True

    # Create a synthetic face image (white background, colored circles at landmark positions)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 200
    import cv2

    # Simulate detected landmarks (shifted/scaled from reference)
    landmarks = np.array(
        [
            [80, 85],  # left eye
            [120, 85],  # right eye
            [100, 110],  # nose
            [85, 135],  # left mouth
            [115, 135],  # right mouth
        ],
        dtype=np.float32,
    )

    # Draw circles at landmark positions
    for lmk in landmarks:
        cv2.circle(img, (int(lmk[0]), int(lmk[1])), 3, (0, 0, 255), -1)

    # Align
    aligned = align_face(img, landmarks, image_size=112)

    p = test_result('Output shape', aligned.shape == (112, 112, 3), f'Got {aligned.shape}')
    passed = passed and p

    p = test_result(
        'Output non-zero', aligned.mean() > 0, f'Mean pixel value: {aligned.mean():.1f}'
    )
    passed = passed and p

    p = test_result('Output is uint8', aligned.dtype == np.uint8, f'Got {aligned.dtype}')
    passed = passed and p

    # Test batch alignment
    from utils.face_align import align_faces_batch, preprocess_for_arcface

    batch = align_faces_batch(img, landmarks[np.newaxis, ...])
    p = test_result('Batch output shape', batch.shape == (1, 112, 112, 3), f'Got {batch.shape}')
    passed = passed and p

    # Test preprocessing
    arcface_input = preprocess_for_arcface(batch)
    p = test_result(
        'ArcFace input shape', arcface_input.shape == (1, 3, 112, 112), f'Got {arcface_input.shape}'
    )
    passed = passed and p

    p = test_result(
        'ArcFace input dtype', arcface_input.dtype == np.float32, f'Got {arcface_input.dtype}'
    )
    passed = passed and p

    # Check normalization range: (pixel - 127.5) / 128.0 -> ~[-1, 1]
    p = test_result(
        'ArcFace normalization range',
        -1.5 < arcface_input.min() < 1.5,
        f'Range: [{arcface_input.min():.3f}, {arcface_input.max():.3f}]',
    )
    return passed and p


def test_api_face_recognize(image_path: Path) -> bool:
    """Test /faces/recognize API endpoint with SCRFD backend."""
    print(f'\n--- API /faces/recognize Test (image: {image_path.name}) ---')

    passed = True

    with open(image_path, 'rb') as f:
        files = {'file': (image_path.name, f, 'image/jpeg')}
        start = time.time()
        try:
            resp = requests.post(
                f'{API_BASE}/v1/faces/recognize',
                files=files,
                timeout=30,
            )
            elapsed = (time.time() - start) * 1000
        except requests.ConnectionError:
            test_result('API reachable', False, 'Connection refused')
            return False

    p = test_result('HTTP status', resp.status_code == 200, f'HTTP {resp.status_code}')
    passed = passed and p

    if resp.status_code != 200:
        print(f'  Response: {resp.text[:200]}')
        return False

    data = resp.json()
    p = test_result('Response time', elapsed < 5000, f'{elapsed:.1f}ms')
    passed = passed and p

    faces = data.get('faces', [])
    num_faces = len(faces)
    p = test_result('Faces detected', num_faces > 0, f'{num_faces} faces')
    passed = passed and p

    if num_faces > 0:
        face = faces[0]

        # Check box
        box = face.get('box', {})
        has_box = all(k in box for k in ('x1', 'y1', 'x2', 'y2'))
        p = test_result('Box present', has_box, str(box))
        passed = passed and p

        # Check confidence
        conf = face.get('confidence', 0)
        p = test_result('Confidence', conf > MIN_CONFIDENCE, f'{conf:.3f}')
        passed = passed and p

        # Check embedding
        embedding = face.get('embedding', [])
        p = test_result(
            'Embedding dimensions', len(embedding) == MIN_EMBEDDING_DIM, f'{len(embedding)} dims'
        )
        passed = passed and p

        if embedding:
            # Check L2 normalization
            norm = np.linalg.norm(embedding)
            p = test_result('Embedding L2 normalized', 0.95 < norm < 1.05, f'L2 norm: {norm:.4f}')
            passed = passed and p

        # Check landmarks (SCRFD should provide these)
        landmarks = face.get('landmarks', [])
        has_landmarks = len(landmarks) == LANDMARK_COUNT
        p = test_result(
            'Landmarks present',
            has_landmarks,
            f'{len(landmarks)} landmarks'
            + (' (SCRFD active)' if has_landmarks else ' (YOLO fallback?)'),
        )
        passed = passed and p

        if has_landmarks:
            # Landmarks should be [0,1] normalized
            lmk_arr = np.array(landmarks)
            in_range = (lmk_arr >= 0).all() and (lmk_arr <= 1).all()
            p = test_result(
                'Landmarks in [0,1]', in_range, f'Range: [{lmk_arr.min():.3f}, {lmk_arr.max():.3f}]'
            )
            passed = passed and p

    return passed


def test_api_face_verify(image_path: Path) -> bool:
    """Test /faces/verify endpoint with the same face (should match)."""
    print('\n--- API /faces/verify Self-Match Test ---')

    with open(image_path, 'rb') as f:
        img_bytes = f.read()

    files = {
        'file1': (image_path.name, img_bytes, 'image/jpeg'),
        'file2': (image_path.name, img_bytes, 'image/jpeg'),
    }
    try:
        resp = requests.post(
            f'{API_BASE}/v1/faces/verify',
            files=files,
            timeout=30,
        )
    except requests.ConnectionError:
        test_result('API reachable', False, 'Connection refused')
        return False

    passed = test_result('HTTP status', resp.status_code == 200, f'HTTP {resp.status_code}')

    if resp.status_code == 200:
        data = resp.json()
        similarity = data.get('similarity', 0)
        matched = data.get('matched', False)

        p = test_result('Self-similarity high', similarity > 0.9, f'Similarity: {similarity:.4f}')
        passed = passed and p

        p = test_result('Self-match', matched)
        passed = passed and p

    return passed


def test_scrfd_benchmark(image_path: Path, iterations: int = 20) -> bool:
    """Benchmark the face recognition pipeline throughput."""
    print(f'\n--- Pipeline Benchmark ({iterations} iterations) ---')

    with open(image_path, 'rb') as f:
        img_bytes = f.read()

    # Warmup
    for _ in range(3):
        requests.post(
            f'{API_BASE}/v1/faces/recognize',
            files={'file': ('test.jpg', img_bytes, 'image/jpeg')},
            timeout=30,
        )

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.time()
        resp = requests.post(
            f'{API_BASE}/v1/faces/recognize',
            files={'file': ('test.jpg', img_bytes, 'image/jpeg')},
            timeout=30,
        )
        elapsed = (time.time() - start) * 1000
        if resp.status_code == 200:
            latencies.append(elapsed)

    if not latencies:
        test_result('Benchmark', False, 'No successful requests')
        return False

    arr = np.array(latencies)
    mean_lat = arr.mean()
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    rps = 1000.0 / mean_lat

    print(f'  Requests: {len(latencies)}/{iterations} successful')
    print(f'  Mean:     {mean_lat:.1f}ms')
    print(f'  P50:      {p50:.1f}ms')
    print(f'  P95:      {p95:.1f}ms')
    print(f'  P99:      {p99:.1f}ms')
    print(f'  RPS:      {rps:.1f} (single-client serial)')

    return test_result('Mean latency < 500ms', mean_lat < 500, f'{mean_lat:.1f}ms')


# =============================================================================
# Main
# =============================================================================


def main():
    print('=' * 60)
    print('SCRFD Face Detection Pipeline Test')
    print('=' * 60)

    results = {}

    # Unit tests (no server needed)
    results['scrfd_decode'] = test_scrfd_decode_unit()
    results['face_alignment'] = test_face_alignment_unit()

    # Integration tests (need running services)
    results['triton_models'] = test_triton_model_ready()

    # Find test image
    test_image = find_test_image()
    if test_image is None:
        print('\nNo test images found. Skipping API tests.')
        print(f'  Expected: {TEST_IMAGE_DIR}/faces/ or {TEST_IMAGE_DIR}/*.jpg')
    else:
        print(f'\nUsing test image: {test_image}')
        results['api_recognize'] = test_api_face_recognize(test_image)
        results['api_verify'] = test_api_face_verify(test_image)
        results['benchmark'] = test_scrfd_benchmark(test_image)

    # Summary
    print('\n' + '=' * 60)
    print('Test Summary')
    print('=' * 60)

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for name, result in results.items():
        status = 'PASS' if result else 'FAIL'
        print(f'  [{status}] {name}')

    print(f'\n  Total: {total}, Passed: {passed}, Failed: {failed}')

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
