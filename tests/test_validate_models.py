#!/usr/bin/env python3
"""
Model output validation against ground truth.

Loads expected_results.json and validates that every modality returns
correct outputs: right classes, correct embedding dimensions, expected
OCR text, and bounding boxes within tolerance.

This is the authoritative test for confirming models are working after
TensorRT export or setup. If this passes, all models are producing
correct inference results.

Usage:
    python tests/test_validate_models.py
    python tests/test_validate_models.py --verbose
    python tests/test_validate_models.py --update   # re-capture ground truth
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import requests


# =============================================================================
# Configuration
# =============================================================================

API_BASE = 'http://localhost:4603'
EXPECTED_FILE = Path(__file__).parent / 'expected_results.json'

# Tolerances — TensorRT rebuilds may shift values slightly
BOX_TOL = 0.03  # normalized coordinate tolerance
CONF_TOL = 0.08  # confidence score tolerance
NORM_TOL = 0.15  # embedding norm tolerance
COSINE_TOL = 0.98  # minimum cosine similarity for embeddings


# =============================================================================
# Utilities
# =============================================================================


class ValidationResult:
    def __init__(self, name: str):
        self.name = name
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, label: str, passed: bool, detail: str = '') -> bool:
        self.checks.append((label, passed, detail))
        return passed

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    @property
    def num_passed(self) -> int:
        return sum(1 for _, ok, _ in self.checks if ok)

    @property
    def num_failed(self) -> int:
        return sum(1 for _, ok, _ in self.checks if not ok)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def boxes_match(expected: list[float], actual: list[float], tol: float = BOX_TOL) -> bool:
    """Check if two bounding boxes are within tolerance."""
    if len(expected) != len(actual):
        return False
    return all(abs(e - a) <= tol for e, a in zip(expected, actual, strict=False))


def print_result(vr: ValidationResult, verbose: bool = False) -> None:
    """Print validation result for a modality."""
    icon = '\033[92m✓\033[0m' if vr.passed else '\033[91m✗\033[0m'
    status = 'PASS' if vr.passed else 'FAIL'
    color = '\033[92m' if vr.passed else '\033[91m'
    reset = '\033[0m'
    print(f'  {icon} {color}{status}{reset}  {vr.name} ({vr.num_passed}/{len(vr.checks)} checks)')

    if verbose or not vr.passed:
        for label, ok, detail in vr.checks:
            check_icon = '  ✓' if ok else '  ✗'
            check_color = '\033[92m' if ok else '\033[91m'
            msg = f'    {check_color}{check_icon}\033[0m {label}'
            if detail:
                msg += f' — {detail}'
            print(msg)


# =============================================================================
# Validation Functions
# =============================================================================


def validate_detection(image_path: str, expected: dict, verbose: bool = False) -> ValidationResult:
    """Validate object detection against ground truth."""
    vr = ValidationResult(f'Detection ({expected["image"]})')

    try:
        with open(image_path, 'rb') as f:
            r = requests.post(f'{API_BASE}/detect', files={'image': f}, timeout=30)
        data = r.json()
    except Exception as e:
        vr.check('API request', False, str(e))
        return vr

    vr.check('API request', r.status_code == 200, f'status={r.status_code}')

    # Check detection count
    num_dets = data.get('num_detections', 0)
    exp_num = expected['num_detections']
    vr.check(
        'Detection count',
        num_dets == exp_num,
        f'got {num_dets}, expected {exp_num}',
    )

    # Check expected classes present
    actual_classes = sorted({d['class_name'] for d in data.get('detections', [])})
    exp_classes = expected['expected_classes']
    vr.check(
        'Expected classes',
        actual_classes == exp_classes,
        f'got {actual_classes}, expected {exp_classes}',
    )

    # Check class counts
    actual_counts: dict[str, int] = {}
    for d in data.get('detections', []):
        cls = d['class_name']
        actual_counts[cls] = actual_counts.get(cls, 0) + 1
    exp_counts = expected['class_counts']
    vr.check(
        'Class counts',
        actual_counts == exp_counts,
        f'got {actual_counts}, expected {exp_counts}',
    )

    # Check each detection box and confidence
    for i, exp_det in enumerate(expected.get('detections', [])):
        if i >= len(data.get('detections', [])):
            vr.check(f'Detection {i}', False, 'missing from output')
            continue

        act_det = data['detections'][i]
        act_box = [round(act_det[k], 3) for k in ('x1', 'y1', 'x2', 'y2')]
        exp_box = exp_det['box']

        vr.check(
            f'Det {i} ({exp_det["class_name"]}) box',
            boxes_match(exp_box, act_box),
            f'got {act_box}, expected {exp_box}',
        )
        vr.check(
            f'Det {i} ({exp_det["class_name"]}) confidence',
            abs(act_det['confidence'] - exp_det['confidence']) <= CONF_TOL,
            f'got {act_det["confidence"]:.3f}, expected {exp_det["confidence"]:.3f}',
        )

    return vr


def validate_faces(expected: dict, verbose: bool = False) -> ValidationResult:
    """Validate face recognition against ground truth."""
    vr = ValidationResult(f'Face Recognition ({expected["image"]})')

    try:
        with open(f'test_images/{expected["image"]}', 'rb') as f:
            r = requests.post(f'{API_BASE}/faces/recognize', files={'image': f}, timeout=30)
        data = r.json()
    except Exception as e:
        vr.check('API request', False, str(e))
        return vr

    vr.check('API request', r.status_code == 200, f'status={r.status_code}')

    # Check face count
    num_faces = data.get('num_faces', 0)
    exp_num = expected['num_faces']
    vr.check('Face count', num_faces == exp_num, f'got {num_faces}, expected {exp_num}')

    faces = data.get('faces', [])
    embeddings = data.get('embeddings', [])

    for i, exp_face in enumerate(expected.get('faces', [])):
        if i >= len(faces):
            vr.check(f'Face {i}', False, 'missing from output')
            continue

        act_face = faces[i]

        # Score
        vr.check(
            f'Face {i} score',
            abs(act_face['score'] - exp_face['score']) <= CONF_TOL,
            f'got {act_face["score"]:.4f}, expected {exp_face["score"]:.4f}',
        )

        # Box
        act_box = [round(v, 4) for v in act_face['box']]
        exp_box = exp_face['box']
        vr.check(
            f'Face {i} box',
            boxes_match(exp_box, act_box),
            f'got {act_box}, expected {exp_box}',
        )

        # Landmarks
        num_lm = len(act_face.get('landmarks', []))
        vr.check(
            f'Face {i} landmarks',
            num_lm == exp_face['num_landmarks'],
            f'got {num_lm} values, expected {exp_face["num_landmarks"]}',
        )

        # Embedding
        if i < len(embeddings):
            emb = embeddings[i]
            vr.check(
                f'Face {i} embedding dim',
                len(emb) == exp_face['embedding_dim'],
                f'got {len(emb)}, expected {exp_face["embedding_dim"]}',
            )
            norm = math.sqrt(sum(x * x for x in emb))
            vr.check(
                f'Face {i} embedding norm',
                abs(norm - exp_face['embedding_norm']) <= NORM_TOL,
                f'got {norm:.4f}, expected {exp_face["embedding_norm"]}',
            )
            # Cosine similarity to expected first-5 values
            if exp_face.get('embedding_first_5'):
                sim = cosine_similarity(emb[:5], exp_face['embedding_first_5'])
                vr.check(
                    f'Face {i} embedding similarity',
                    sim >= COSINE_TOL,
                    f'cosine_sim={sim:.4f} (threshold={COSINE_TOL})',
                )

    return vr


def validate_clip_image(expected: dict, verbose: bool = False) -> ValidationResult:
    """Validate CLIP image embedding against ground truth."""
    vr = ValidationResult(f'CLIP Image Embedding ({expected["image"]})')

    try:
        with open(f'test_images/{expected["image"]}', 'rb') as f:
            r = requests.post(f'{API_BASE}/embed/image', files={'image': f}, timeout=30)
        data = r.json()
    except Exception as e:
        vr.check('API request', False, str(e))
        return vr

    vr.check('API request', r.status_code == 200, f'status={r.status_code}')

    emb = data.get('embedding', [])

    # Dimensions
    vr.check(
        'Embedding dim',
        len(emb) == expected['dimensions'],
        f'got {len(emb)}, expected {expected["dimensions"]}',
    )

    # Norm
    norm = math.sqrt(sum(x * x for x in emb))
    vr.check(
        'Embedding norm',
        abs(norm - expected['norm']) <= NORM_TOL,
        f'got {norm:.4f}, expected {expected["norm"]}',
    )

    # Cosine similarity to expected embedding
    if expected.get('first_5') and len(emb) >= 5:
        sim = cosine_similarity(emb[:5], expected['first_5'])
        vr.check(
            'Embedding similarity (first 5)',
            sim >= COSINE_TOL,
            f'cosine_sim={sim:.4f} (threshold={COSINE_TOL})',
        )
    if expected.get('last_5') and len(emb) >= 5:
        sim = cosine_similarity(emb[-5:], expected['last_5'])
        vr.check(
            'Embedding similarity (last 5)',
            sim >= COSINE_TOL,
            f'cosine_sim={sim:.4f} (threshold={COSINE_TOL})',
        )

    return vr


def validate_clip_text(expected: dict, verbose: bool = False) -> ValidationResult:
    """Validate CLIP text embedding against ground truth."""
    vr = ValidationResult(f'CLIP Text Embedding ("{expected["text"]}")')

    try:
        r = requests.post(f'{API_BASE}/embed/text', json={'text': expected['text']}, timeout=30)
        data = r.json()
    except Exception as e:
        vr.check('API request', False, str(e))
        return vr

    vr.check('API request', r.status_code == 200, f'status={r.status_code}')

    emb = data.get('embedding', [])

    vr.check(
        'Embedding dim',
        len(emb) == expected['dimensions'],
        f'got {len(emb)}, expected {expected["dimensions"]}',
    )

    norm = math.sqrt(sum(x * x for x in emb))
    vr.check(
        'Embedding norm',
        abs(norm - expected['norm']) <= NORM_TOL,
        f'got {norm:.4f}, expected {expected["norm"]}',
    )

    if expected.get('first_5') and len(emb) >= 5:
        sim = cosine_similarity(emb[:5], expected['first_5'])
        vr.check(
            'Embedding similarity (first 5)',
            sim >= COSINE_TOL,
            f'cosine_sim={sim:.4f} (threshold={COSINE_TOL})',
        )

    return vr


def validate_ocr(expected: dict, verbose: bool = False) -> ValidationResult:
    """Validate OCR output against ground truth."""
    vr = ValidationResult(f'OCR ({expected["image"]})')

    try:
        with open(f'test_images/{expected["image"]}', 'rb') as f:
            r = requests.post(f'{API_BASE}/ocr/predict', files={'image': f}, timeout=30)
        data = r.json()
    except Exception as e:
        vr.check('API request', False, str(e))
        return vr

    vr.check('API request', r.status_code == 200, f'status={r.status_code}')

    # Text count
    num_texts = data.get('num_texts', 0)
    vr.check(
        'Text count',
        num_texts == expected['num_texts'],
        f'got {num_texts}, expected {expected["num_texts"]}',
    )

    # Exact text match (case-insensitive for OCR tolerance)
    actual_texts = data.get('texts', [])
    exp_texts = expected['texts']
    for i, exp_text in enumerate(exp_texts):
        if i < len(actual_texts):
            # OCR can have minor character variations, use case-insensitive contains
            match = actual_texts[i].lower() == exp_text.lower()
            vr.check(
                f'Text {i}',
                match,
                f'got "{actual_texts[i]}", expected "{exp_text}"',
            )
        else:
            vr.check(f'Text {i}', False, f'missing, expected "{exp_text}"')

    # Full text
    actual_full = data.get('full_text', '')
    exp_full = expected['full_text']
    vr.check(
        'Full text',
        actual_full.lower() == exp_full.lower(),
        f'got "{actual_full}", expected "{exp_full}"',
    )

    # Region detection scores
    for i, _exp_reg in enumerate(expected.get('regions', [])):
        regions = data.get('regions', [])
        if i < len(regions):
            act_reg = regions[i]
            vr.check(
                f'Region {i} det_score',
                act_reg.get('det_score', 0) >= 0.5,
                f'got {act_reg.get("det_score", 0):.4f}',
            )
            vr.check(
                f'Region {i} rec_score',
                act_reg.get('rec_score', 0) >= 0.5,
                f'got {act_reg.get("rec_score", 0):.4f}',
            )

    return vr


def validate_analyze(expected: dict, verbose: bool = False) -> ValidationResult:
    """Validate combined analysis against ground truth."""
    vr = ValidationResult(f'Analyze ({expected["image"]})')

    try:
        with open(f'test_images/{expected["image"]}', 'rb') as f:
            r = requests.post(f'{API_BASE}/analyze', files={'image': f}, timeout=60)
        data = r.json()
    except Exception as e:
        vr.check('API request', False, str(e))
        return vr

    vr.check('API request', r.status_code == 200, f'status={r.status_code}')
    vr.check('Status', data.get('status') == 'success', f'got {data.get("status")}')

    # Detection count
    num_dets = data.get('num_detections', 0)
    vr.check(
        'Detection count',
        num_dets == expected['num_detections'],
        f'got {num_dets}, expected {expected["num_detections"]}',
    )

    # Detection classes
    actual_classes = sorted({d.get('class_name', '') for d in data.get('detections', [])})
    vr.check(
        'Detection classes',
        actual_classes == expected['detection_classes'],
        f'got {actual_classes}, expected {expected["detection_classes"]}',
    )

    # Face count
    num_faces = data.get('num_faces', 0)
    vr.check(
        'Face count',
        num_faces == expected['num_faces'],
        f'got {num_faces}, expected {expected["num_faces"]}',
    )

    # OCR
    ocr = data.get('ocr', {})
    ocr_count = ocr.get('num_texts', 0) if ocr else 0
    vr.check(
        'OCR text count',
        ocr_count == expected['ocr_num_texts'],
        f'got {ocr_count}, expected {expected["ocr_num_texts"]}',
    )

    if expected.get('ocr_texts'):
        actual_ocr = [t for t in ocr.get('texts', []) if t] if ocr else []
        for exp_text in expected['ocr_texts']:
            found = any(exp_text.lower() in t.lower() for t in actual_ocr)
            vr.check(f'OCR contains "{exp_text}"', found, f'in {actual_ocr}')

    return vr


# =============================================================================
# Update Ground Truth
# =============================================================================


def update_ground_truth() -> None:
    """Re-capture ground truth from live API and save to expected_results.json."""
    print('Capturing ground truth from live API...\n')

    ground_truth: dict = {
        '_description': 'Ground truth expected outputs for validating all model modalities.',
        '_usage': 'Run: python tests/test_validate_models.py',
        '_images': {
            'bus.jpg': '810x1080 Ultralytics standard test image (bus, people, text)',
            'zidane.jpg': '1280x720 Ultralytics standard test image (2 people, 2 ties)',
        },
        '_tolerances': {
            'box_coords': BOX_TOL,
            'confidence': CONF_TOL,
            'embedding_norm': NORM_TOL,
            'embedding_cosine_sim': COSINE_TOL,
        },
    }

    # Detection - bus.jpg
    print('  1/7 Detection (bus.jpg)...')
    with open('test_images/bus.jpg', 'rb') as img:
        r = requests.post(f'{API_BASE}/detect', files={'image': img}, timeout=30)
    data = r.json()
    class_counts: dict[str, int] = {}
    for d in data['detections']:
        cls = d['class_name']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    ground_truth['detect_bus'] = {
        'image': 'bus.jpg',
        'image_size': data['image'],
        'num_detections': data['num_detections'],
        'expected_classes': sorted(class_counts.keys()),
        'class_counts': class_counts,
        'detections': [
            {
                'class_name': d['class_name'],
                'class_id': d['class'],
                'confidence': round(d['confidence'], 3),
                'box': [round(d[k], 3) for k in ('x1', 'y1', 'x2', 'y2')],
            }
            for d in data['detections']
        ],
    }

    # Detection - zidane.jpg
    print('  2/7 Detection (zidane.jpg)...')
    with open('test_images/zidane.jpg', 'rb') as img:
        r = requests.post(f'{API_BASE}/detect', files={'image': img}, timeout=30)
    data = r.json()
    class_counts = {}
    for d in data['detections']:
        cls = d['class_name']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    ground_truth['detect_zidane'] = {
        'image': 'zidane.jpg',
        'image_size': data['image'],
        'num_detections': data['num_detections'],
        'expected_classes': sorted(class_counts.keys()),
        'class_counts': class_counts,
        'detections': [
            {
                'class_name': d['class_name'],
                'class_id': d['class'],
                'confidence': round(d['confidence'], 3),
                'box': [round(d[k], 3) for k in ('x1', 'y1', 'x2', 'y2')],
            }
            for d in data['detections']
        ],
    }

    # Face recognition - zidane.jpg
    print('  3/7 Face recognition (zidane.jpg)...')
    with open('test_images/zidane.jpg', 'rb') as img:
        r = requests.post(f'{API_BASE}/faces/recognize', files={'image': img}, timeout=30)
    data = r.json()
    faces_out = []
    for i, f in enumerate(data.get('faces', [])):
        emb = data.get('embeddings', [[]])[i] if i < len(data.get('embeddings', [])) else []
        norm = math.sqrt(sum(x * x for x in emb)) if emb else 0
        faces_out.append(
            {
                'score': round(f['score'], 4),
                'box': [round(v, 4) for v in f['box']],
                'num_landmarks': len(f.get('landmarks', [])),
                'embedding_dim': len(emb),
                'embedding_norm': round(norm, 4),
                'embedding_first_5': [round(x, 6) for x in emb[:5]],
            }
        )
    ground_truth['faces_zidane'] = {
        'image': 'zidane.jpg',
        'num_faces': data['num_faces'],
        'faces': faces_out,
    }

    # CLIP image embedding - bus.jpg
    print('  4/7 CLIP image embedding (bus.jpg)...')
    with open('test_images/bus.jpg', 'rb') as img:
        r = requests.post(f'{API_BASE}/embed/image', files={'image': img}, timeout=30)
    data = r.json()
    emb = data['embedding']
    norm = math.sqrt(sum(x * x for x in emb))
    ground_truth['embed_image_bus'] = {
        'image': 'bus.jpg',
        'dimensions': len(emb),
        'norm': round(norm, 4),
        'first_5': [round(x, 6) for x in emb[:5]],
        'last_5': [round(x, 6) for x in emb[-5:]],
    }

    # CLIP text embedding
    print('  5/7 CLIP text embedding...')
    r = requests.post(f'{API_BASE}/embed/text', json={'text': 'a bus on the street'}, timeout=30)
    data = r.json()
    emb = data['embedding']
    norm = math.sqrt(sum(x * x for x in emb))
    ground_truth['embed_text'] = {
        'text': 'a bus on the street',
        'dimensions': len(emb),
        'norm': round(norm, 4),
        'first_5': [round(x, 6) for x in emb[:5]],
        'last_5': [round(x, 6) for x in emb[-5:]],
    }

    # OCR - bus.jpg
    print('  6/7 OCR (bus.jpg)...')
    with open('test_images/bus.jpg', 'rb') as img:
        r = requests.post(f'{API_BASE}/ocr/predict', files={'image': img}, timeout=30)
    data = r.json()
    ground_truth['ocr_bus'] = {
        'image': 'bus.jpg',
        'num_texts': data['num_texts'],
        'texts': data['texts'],
        'full_text': data['full_text'],
        'regions': [
            {
                'text': reg['text'],
                'det_score': round(reg['det_score'], 4),
                'rec_score': round(reg['rec_score'], 4),
                'box_normalized': [round(v, 4) for v in reg['box_normalized']],
            }
            for reg in data['regions']
        ],
    }

    # Analyze - bus.jpg
    print('  7/7 Analyze (bus.jpg)...')
    with open('test_images/bus.jpg', 'rb') as img:
        r = requests.post(f'{API_BASE}/analyze', files={'image': img}, timeout=60)
    data = r.json()
    det_classes = sorted({d.get('class_name', '') for d in data.get('detections', [])})
    ocr = data.get('ocr', {})
    ground_truth['analyze_bus'] = {
        'image': 'bus.jpg',
        'status': data.get('status'),
        'num_detections': data.get('num_detections'),
        'detection_classes': det_classes,
        'num_faces': data.get('num_faces'),
        'has_global_embedding': data.get('global_embedding') is not None,
        'ocr_num_texts': ocr.get('num_texts', 0) if ocr else 0,
        'ocr_texts': [t for t in ocr.get('texts', []) if t] if ocr else [],
    }

    with open(EXPECTED_FILE, 'w') as f:
        json.dump(ground_truth, f, indent=2)
        f.write('\n')

    print(f'\nGround truth saved to {EXPECTED_FILE}')


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate model outputs against ground truth')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all checks')
    parser.add_argument(
        '--update', action='store_true', help='Re-capture ground truth from live API'
    )
    args = parser.parse_args()

    if args.update:
        update_ground_truth()
        return 0

    # Load ground truth
    if not EXPECTED_FILE.exists():
        print(f'Ground truth file not found: {EXPECTED_FILE}')
        print('Run with --update to capture ground truth first.')
        return 1

    with open(EXPECTED_FILE) as f:
        expected = json.load(f)

    print()
    print('=' * 70)
    print('MODEL OUTPUT VALIDATION'.center(70))
    print('=' * 70)
    print()

    all_results: list[ValidationResult] = []

    # 1. Detection - bus.jpg
    vr = validate_detection('test_images/bus.jpg', expected['detect_bus'], args.verbose)
    print_result(vr, args.verbose)
    all_results.append(vr)

    # 2. Detection - zidane.jpg
    vr = validate_detection('test_images/zidane.jpg', expected['detect_zidane'], args.verbose)
    print_result(vr, args.verbose)
    all_results.append(vr)

    # 3. Face recognition - zidane.jpg
    vr = validate_faces(expected['faces_zidane'], args.verbose)
    print_result(vr, args.verbose)
    all_results.append(vr)

    # 4. CLIP image embedding - bus.jpg
    vr = validate_clip_image(expected['embed_image_bus'], args.verbose)
    print_result(vr, args.verbose)
    all_results.append(vr)

    # 5. CLIP text embedding
    vr = validate_clip_text(expected['embed_text'], args.verbose)
    print_result(vr, args.verbose)
    all_results.append(vr)

    # 6. OCR - bus.jpg
    vr = validate_ocr(expected['ocr_bus'], args.verbose)
    print_result(vr, args.verbose)
    all_results.append(vr)

    # 7. Analyze - bus.jpg
    vr = validate_analyze(expected['analyze_bus'], args.verbose)
    print_result(vr, args.verbose)
    all_results.append(vr)

    # Summary
    total_checks = sum(len(vr.checks) for vr in all_results)
    passed_checks = sum(vr.num_passed for vr in all_results)
    failed_modalities = sum(1 for vr in all_results if not vr.passed)
    total_modalities = len(all_results)

    print()
    print('-' * 70)
    print(
        f'  Modalities: {total_modalities - failed_modalities}/{total_modalities} passed'
        f'    Checks: {passed_checks}/{total_checks} passed'
    )

    if failed_modalities == 0:
        print('\n  All models producing correct outputs.')
        return 0

    print(f'\n  {failed_modalities} modality(ies) failed. Run with --verbose for details.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
