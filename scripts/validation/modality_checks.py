"""Per-modality validation functions: each calls the live API and compares to ground truth."""

from __future__ import annotations

import math
from pathlib import Path

import requests

from scripts.validation.validate_models_lib import (
    API_BASE,
    CONF_TOL,
    COSINE_TOL,
    NORM_TOL,
    ValidationResult,
    boxes_match,
    cosine_similarity,
)


def validate_detection(image_path: str, expected: dict) -> ValidationResult:
    """Validate object detection against ground truth."""
    vr = ValidationResult(f'Detection ({expected["image"]})')

    try:
        with Path(image_path).open('rb') as f:
            r = requests.post(f'{API_BASE}/detect', files={'image': f}, timeout=30)
        data = r.json()
    except (requests.RequestException, OSError, ValueError) as e:
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


def validate_faces(expected: dict) -> ValidationResult:
    """Validate face recognition against ground truth."""
    vr = ValidationResult(f'Face Recognition ({expected["image"]})')

    try:
        with Path(f'test_images/{expected["image"]}').open('rb') as f:
            r = requests.post(f'{API_BASE}/faces/recognize', files={'image': f}, timeout=30)
        data = r.json()
    except (requests.RequestException, OSError, ValueError) as e:
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


def validate_clip_image(expected: dict) -> ValidationResult:
    """Validate CLIP image embedding against ground truth."""
    vr = ValidationResult(f'CLIP Image Embedding ({expected["image"]})')

    try:
        with Path(f'test_images/{expected["image"]}').open('rb') as f:
            r = requests.post(f'{API_BASE}/embed/image', files={'image': f}, timeout=30)
        data = r.json()
    except (requests.RequestException, OSError, ValueError) as e:
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


def validate_clip_text(expected: dict) -> ValidationResult:
    """Validate CLIP text embedding against ground truth."""
    vr = ValidationResult(f'CLIP Text Embedding ("{expected["text"]}")')

    try:
        r = requests.post(f'{API_BASE}/embed/text', json={'text': expected['text']}, timeout=30)
        data = r.json()
    except (requests.RequestException, OSError, ValueError) as e:
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


def validate_ocr(expected: dict) -> ValidationResult:
    """Validate OCR output against ground truth."""
    vr = ValidationResult(f'OCR ({expected["image"]})')

    try:
        with Path(f'test_images/{expected["image"]}').open('rb') as f:
            r = requests.post(f'{API_BASE}/ocr/predict', files={'image': f}, timeout=30)
        data = r.json()
    except (requests.RequestException, OSError, ValueError) as e:
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


def validate_analyze(expected: dict) -> ValidationResult:
    """Validate combined analysis against ground truth."""
    vr = ValidationResult(f'Analyze ({expected["image"]})')

    try:
        with Path(f'test_images/{expected["image"]}').open('rb') as f:
            r = requests.post(f'{API_BASE}/analyze', files={'image': f}, timeout=60)
        data = r.json()
    except (requests.RequestException, OSError, ValueError) as e:
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
