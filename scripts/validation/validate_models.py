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
    python scripts/validation/validate_models.py
    python scripts/validation/validate_models.py --verbose
    python scripts/validation/validate_models.py --update   # re-capture ground truth
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import requests


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.validation.modality_checks import (
    validate_analyze,
    validate_clip_image,
    validate_clip_text,
    validate_detection,
    validate_faces,
    validate_ocr,
)
from scripts.validation.validate_models_lib import (
    API_BASE,
    BOX_TOL,
    CONF_TOL,
    COSINE_TOL,
    EXPECTED_FILE,
    NORM_TOL,
    ValidationResult,
    print_result,
)


def update_ground_truth() -> None:
    """Re-capture ground truth from live API and save to expected_results.json."""
    print('Capturing ground truth from live API...\n')

    ground_truth: dict = {
        '_description': 'Ground truth expected outputs for validating all model modalities.',
        '_usage': 'Run: python scripts/validation/validate_models.py',
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
    with Path('test_images/bus.jpg').open('rb') as img:
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
    with Path('test_images/zidane.jpg').open('rb') as img:
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
    with Path('test_images/zidane.jpg').open('rb') as img:
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
    with Path('test_images/bus.jpg').open('rb') as img:
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
    with Path('test_images/bus.jpg').open('rb') as img:
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
    with Path('test_images/bus.jpg').open('rb') as img:
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

    with Path(EXPECTED_FILE).open('w') as f:
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

    with Path(EXPECTED_FILE).open() as f:
        expected = json.load(f)

    print()
    print('=' * 70)
    print('MODEL OUTPUT VALIDATION'.center(70))
    print('=' * 70)
    print()

    all_results: list[ValidationResult] = []

    # 1. Detection - bus.jpg
    vr = validate_detection('test_images/bus.jpg', expected['detect_bus'])
    print_result(vr, args.verbose)
    all_results.append(vr)

    # 2. Detection - zidane.jpg
    vr = validate_detection('test_images/zidane.jpg', expected['detect_zidane'])
    print_result(vr, args.verbose)
    all_results.append(vr)

    # 3. Face recognition - zidane.jpg
    vr = validate_faces(expected['faces_zidane'])
    print_result(vr, args.verbose)
    all_results.append(vr)

    # 4. CLIP image embedding - bus.jpg
    vr = validate_clip_image(expected['embed_image_bus'])
    print_result(vr, args.verbose)
    all_results.append(vr)

    # 5. CLIP text embedding
    vr = validate_clip_text(expected['embed_text'])
    print_result(vr, args.verbose)
    all_results.append(vr)

    # 6. OCR - bus.jpg
    vr = validate_ocr(expected['ocr_bus'])
    print_result(vr, args.verbose)
    all_results.append(vr)

    # 7. Analyze - bus.jpg
    vr = validate_analyze(expected['analyze_bus'])
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
