#!/usr/bin/env python3
"""
Visual validation script - draws bounding boxes and saves annotated images.

Tests detection, face detection, and OCR results by drawing them on images.

Usage:
    python tests/validate_visual_results.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import requests


# =============================================================================
# Configuration
# =============================================================================

API_BASE = 'http://localhost:4603'
TEST_IMAGE = Path('test_images/zidane.jpg')
OCR_IMAGE = Path('test_images/ocr-real/testocr_rgb.jpg')
OUTPUT_DIR = Path('test_results')

# Colors (BGR format for OpenCV)
COLORS = {
    'person': (0, 255, 0),  # Green
    'face': (255, 0, 0),  # Blue
    'text': (0, 0, 255),  # Red
    'default': (255, 255, 0),  # Cyan
}


# =============================================================================
# Drawing Functions
# =============================================================================


def draw_detection_boxes(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw object detection bounding boxes."""
    h, w = image.shape[:2]

    for det in detections:
        # Convert normalized coordinates to pixel coordinates
        # Handle both formats: {x1, y1, x2, y2} from /detect and {box: [x1, y1, x2, y2]} from /analyze
        if 'box' in det and isinstance(det['box'], list):
            x1 = int(det['box'][0] * w)
            y1 = int(det['box'][1] * h)
            x2 = int(det['box'][2] * w)
            y2 = int(det['box'][3] * h)
        else:
            x1 = int(det['x1'] * w)
            y1 = int(det['y1'] * h)
            x2 = int(det['x2'] * w)
            y2 = int(det['y2'] * h)

        class_name = det.get('class_name', det.get('class', 'unknown'))
        confidence = det.get('confidence', 0.0)

        # Draw box
        color = COLORS.get(class_name, COLORS['default'])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f'{class_name}: {confidence:.2f}'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image


def draw_face_boxes(image: np.ndarray, faces: list) -> np.ndarray:
    """Draw face detection boxes with landmarks."""
    h, w = image.shape[:2]

    for i, face in enumerate(faces):
        box = face.get('box', {})
        if isinstance(box, list) and len(box) >= 4:
            # Box is a list [x1, y1, x2, y2]
            box = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}
        landmarks = face.get('landmarks', [])
        confidence = face.get('score', face.get('confidence', 0.0))

        # Convert normalized coordinates to pixel coordinates
        x1 = int(box['x1'] * w)
        y1 = int(box['y1'] * h)
        x2 = int(box['x2'] * w)
        y2 = int(box['y2'] * h)

        # Draw face box
        cv2.rectangle(image, (x1, y1), (x2, y2), COLORS['face'], 2)

        # Draw face ID label
        label = f'Face {i + 1}: {confidence:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['face'], 2)

        # Draw landmarks (5 points: left eye, right eye, nose, left mouth, right mouth)
        # Landmarks are a flat list [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
        if landmarks and len(landmarks) >= 10:
            for lm_idx in range(0, 10, 2):
                lm_x = int(landmarks[lm_idx] * w)
                lm_y = int(landmarks[lm_idx + 1] * h)
                if lm_x > 0 and lm_y > 0:  # Skip invalid landmarks (0, 0)
                    cv2.circle(image, (lm_x, lm_y), 3, (0, 255, 255), -1)

    return image


def draw_ocr_boxes(image: np.ndarray, regions: list) -> np.ndarray:
    """Draw OCR text regions."""
    h, w = image.shape[:2]

    for region in regions:
        box = region.get('box', [])
        text = region.get('text', '')
        confidence = region.get('confidence', 0.0)

        if len(box) >= 4:
            # Convert normalized coordinates to pixel coordinates
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            # Draw text box
            cv2.rectangle(image, (x1, y1), (x2, y2), COLORS['text'], 2)

            # Draw text label (truncate if too long)
            label = text[:30] + '...' if len(text) > 30 else text
            label = f'{label} ({confidence:.2f})'

            cv2.putText(
                image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1
            )

    return image


# =============================================================================
# Test Functions
# =============================================================================


def test_detection_visual() -> bool:
    """Test and visualize object detection."""
    print('\n[1/4] Testing Object Detection...')

    if not TEST_IMAGE.exists():
        print(f'  ❌ Test image not found: {TEST_IMAGE}')
        return False

    # Load image
    image = cv2.imread(str(TEST_IMAGE))
    if image is None:
        print(f'  ❌ Could not load image: {TEST_IMAGE}')
        return False

    # Call detection API
    with open(TEST_IMAGE, 'rb') as f:
        files = {'image': f}
        response = requests.post(f'{API_BASE}/detect', files=files, timeout=30)

    if response.status_code != 200:
        print(f'  ❌ API error: {response.status_code}')
        return False

    data = response.json()
    detections = data.get('detections', [])

    print(f'  ✓ Detected {len(detections)} objects')
    for det in detections:
        print(f'    - {det["class_name"]}: {det["confidence"]:.2f}')

    # Draw boxes
    image_with_boxes = draw_detection_boxes(image.copy(), detections)

    # Save result
    output_path = OUTPUT_DIR / 'detection_result.jpg'
    cv2.imwrite(str(output_path), image_with_boxes)
    print(f'  ✓ Saved to: {output_path}')

    return True


def test_faces_visual() -> bool:
    """Test and visualize face detection."""
    print('\n[2/4] Testing Face Detection...')

    if not TEST_IMAGE.exists():
        print(f'  ❌ Test image not found: {TEST_IMAGE}')
        return False

    # Load image
    image = cv2.imread(str(TEST_IMAGE))
    if image is None:
        print(f'  ❌ Could not load image: {TEST_IMAGE}')
        return False

    # Call face detection API
    with open(TEST_IMAGE, 'rb') as f:
        files = {'image': f}
        response = requests.post(f'{API_BASE}/faces/detect', files=files, timeout=30)

    if response.status_code != 200:
        print(f'  ❌ API error: {response.status_code}')
        return False

    data = response.json()
    faces = data.get('faces', [])

    print(f'  ✓ Detected {len(faces)} faces')
    for i, face in enumerate(faces):
        score = face.get('score', face.get('confidence', 0.0))
        print(f'    - Face {i + 1}: confidence={score:.2f}')

    # Draw boxes
    image_with_boxes = draw_face_boxes(image.copy(), faces)

    # Save result
    output_path = OUTPUT_DIR / 'faces_result.jpg'
    cv2.imwrite(str(output_path), image_with_boxes)
    print(f'  ✓ Saved to: {output_path}')

    return True


def test_ocr_visual() -> bool:
    """Test and visualize OCR."""
    print('\n[3/4] Testing OCR...')

    # Try to find OCR test image
    ocr_images = list(Path('test_images').glob('**/testocr*.jpg'))
    if not ocr_images:
        ocr_images = [TEST_IMAGE]

    test_img = ocr_images[0]
    if not test_img.exists():
        print(f'  ❌ Test image not found: {test_img}')
        return False

    # Load image
    image = cv2.imread(str(test_img))
    if image is None:
        print(f'  ❌ Could not load image: {test_img}')
        return False

    # Call OCR API
    with open(test_img, 'rb') as f:
        files = {'image': f}
        response = requests.post(f'{API_BASE}/ocr/predict', files=files, timeout=30)

    if response.status_code != 200:
        print(f'  ❌ API error: {response.status_code}')
        return False

    data = response.json()
    regions = data.get('regions', [])

    print(f'  ✓ Detected {len(regions)} text regions')
    for i, region in enumerate(regions[:5]):  # Show first 5
        text = region.get('text', '')
        print(f'    - Text {i + 1}: "{text[:50]}"')

    # Draw boxes
    image_with_boxes = draw_ocr_boxes(image.copy(), regions)

    # Save result
    output_path = OUTPUT_DIR / 'ocr_result.jpg'
    cv2.imwrite(str(output_path), image_with_boxes)
    print(f'  ✓ Saved to: {output_path}')

    return True


def test_analyze_visual() -> bool:
    """Test and visualize combined analysis."""
    print('\n[4/4] Testing Combined Analysis...')

    if not TEST_IMAGE.exists():
        print(f'  ❌ Test image not found: {TEST_IMAGE}')
        return False

    # Load image
    image = cv2.imread(str(TEST_IMAGE))
    if image is None:
        print(f'  ❌ Could not load image: {TEST_IMAGE}')
        return False

    # Call analyze API
    with open(TEST_IMAGE, 'rb') as f:
        files = {'image': f}
        response = requests.post(f'{API_BASE}/analyze', files=files, timeout=30)

    if response.status_code != 200:
        print(f'  ❌ API error: {response.status_code}')
        return False

    data = response.json()

    print('  ✓ Analysis complete:')
    print(f'    - Objects: {data.get("num_detections", 0)}')
    print(f'    - Faces: {data.get("num_faces", 0)}')
    print(f'    - Text: {"yes" if data.get("has_text") else "no"}')
    embedding = data.get('global_embedding')
    if embedding:
        print(f'    - Embedding: {len(embedding)} dims')
    else:
        print('    - Embedding: not included')

    # Draw all annotations
    result = image.copy()
    if 'detections' in data:
        result = draw_detection_boxes(result, data['detections'])
    if 'faces' in data:
        result = draw_face_boxes(result, data['faces'])

    # Save result
    output_path = OUTPUT_DIR / 'analyze_result.jpg'
    cv2.imwrite(str(output_path), result)
    print(f'  ✓ Saved to: {output_path}')

    return True


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Main entry point."""
    print('=' * 80)
    print('VISUAL VALIDATION - Bounding Box Verification'.center(80))
    print('=' * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run all visual tests
    results = []
    results.append(test_detection_visual())
    results.append(test_faces_visual())
    results.append(test_ocr_visual())
    results.append(test_analyze_visual())

    # Summary
    print('\n' + '=' * 80)
    print('SUMMARY'.center(80))
    print('=' * 80)
    passed = sum(results)
    total = len(results)
    print(f'\n  Tests: {passed}/{total} passed')
    print(f'  Output: {OUTPUT_DIR.absolute()}/')
    print('\n  Annotated images:')
    for img in OUTPUT_DIR.glob('*.jpg'):
        print(f'    - {img.name}')

    if passed == total:
        print('\n✅ All visual validations passed!')
        print('   Review the annotated images to verify bounding boxes are correct.')
        return 0

    print('\n❌ Some validations failed.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
