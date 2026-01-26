#!/usr/bin/env python3
"""
Comprehensive system test suite for Triton API.

Tests all core functionality:
- Service health (API, Triton, OpenSearch)
- Individual model endpoints
- Full ingest pipeline
- Search and query functionality
- OpenSearch indexes

Usage:
    python tests/test_full_system.py
    python tests/test_full_system.py --skip-ingest  # Skip slow ingestion tests
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


# =============================================================================
# Configuration
# =============================================================================

API_BASE = 'http://localhost:4603'
TRITON_BASE = 'http://localhost:4600'
OPENSEARCH_BASE = 'http://localhost:4607'

TEST_IMAGE = Path('test_images/zidane.jpg')
TEST_IMAGE_2 = Path('test_images/bus.jpg')
INGEST_DIR = Path('test_images/ingest_test_50')

# Results tracking
results: dict[str, Any] = {
    'passed': 0,
    'failed': 0,
    'skipped': 0,
    'tests': [],
}


# =============================================================================
# Test Utilities
# =============================================================================


def print_header(text: str) -> None:
    """Print a section header."""
    print(f'\n{"=" * 80}')
    print(f'{text.center(80)}')
    print('=' * 80)


def print_test(name: str, status: str, message: str = '') -> None:
    """Print a test result."""
    icons = {'PASS': '✓', 'FAIL': '✗', 'SKIP': '○'}
    colors = {'PASS': '\033[92m', 'FAIL': '\033[91m', 'SKIP': '\033[93m'}
    reset = '\033[0m'

    icon = icons.get(status, '?')
    color = colors.get(status, '')
    status_text = f'{color}{icon} {status}{reset}'

    msg_text = f' - {message}' if message else ''
    print(f'  {status_text:<20} {name}{msg_text}')

    results['tests'].append({'name': name, 'status': status, 'message': message})
    if status == 'PASS':
        results['passed'] += 1
    elif status == 'FAIL':
        results['failed'] += 1
    else:
        results['skipped'] += 1


def test_endpoint(name: str, method: str, url: str, **kwargs) -> dict[str, Any] | None:
    """Test an API endpoint and report results."""
    # Extract timeout from kwargs or use default
    timeout = kwargs.pop('timeout', 30)

    try:
        if method == 'GET':
            response = requests.get(url, timeout=timeout, **kwargs)
        elif method == 'POST':
            response = requests.post(url, timeout=timeout, **kwargs)
        elif method == 'DELETE':
            response = requests.delete(url, timeout=timeout, **kwargs)
        else:
            print_test(name, 'FAIL', f'Unknown method: {method}')
            return None

        if response.status_code in [200, 201]:  # Accept 200 OK and 201 Created
            # Some endpoints return empty body (e.g. Triton health)
            if response.text:
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    # Non-JSON response but still successful
                    print_test(name, 'PASS', f'{response.elapsed.total_seconds():.3f}s')
                    return {}
            else:
                data = {}
            print_test(name, 'PASS', f'{response.elapsed.total_seconds():.3f}s')
            return data

        print_test(name, 'FAIL', f'Status {response.status_code}: {response.text[:100]}')
        return None

    except requests.exceptions.ConnectionError:
        print_test(name, 'FAIL', 'Connection refused - is the service running?')
        return None
    except requests.exceptions.Timeout:
        print_test(name, 'FAIL', 'Request timeout (>30s)')
        return None
    except Exception as e:
        print_test(name, 'FAIL', f'Error: {e}')
        return None


# =============================================================================
# Health Checks
# =============================================================================


def clear_opensearch_data() -> bool:
    """Clear all OpenSearch visual_search indexes for fresh testing."""
    print_header('CLEARING OPENSEARCH DATA')

    try:
        response = requests.delete(f'{OPENSEARCH_BASE}/visual_search_*', timeout=10)
        if response.status_code in [200, 404]:
            print_test('Clear OpenSearch Indexes', 'PASS', 'All visual_search_* indexes cleared')
            return True
        print_test('Clear OpenSearch Indexes', 'FAIL', f'Status {response.status_code}')
        return False
    except Exception as e:
        print_test('Clear OpenSearch Indexes', 'FAIL', f'Error: {e}')
        return False


def test_health_checks() -> bool:
    """Test all service health endpoints."""
    print_header('SERVICE HEALTH CHECKS')

    # API health
    data = test_endpoint('API Health', 'GET', f'{API_BASE}/health')
    if not data:
        return False

    # Triton health (returns empty body on success)
    data = test_endpoint('Triton Health', 'GET', f'{TRITON_BASE}/v2/health/ready')
    if data is None:  # None means failure, {} means success with no body
        return False

    # OpenSearch health
    data = test_endpoint('OpenSearch Health', 'GET', f'{OPENSEARCH_BASE}/_cluster/health')
    if data and data.get('status') in ['green', 'yellow']:
        print_test('OpenSearch Status', 'PASS', f'Status: {data["status"]}')
    else:
        print_test('OpenSearch Status', 'FAIL', 'Cluster not healthy')
        return False

    # List Triton models
    response = requests.post(f'{TRITON_BASE}/v2/repository/index', timeout=10)
    if response.status_code == 200:
        models = response.json()
        ready_models = [m['name'] for m in models if m.get('state') == 'READY']
        print_test(
            'Triton Models',
            'PASS',
            f'{len(ready_models)} models ready: {", ".join(ready_models[:3])}...',
        )
    else:
        print_test('Triton Models', 'FAIL', 'Could not list models')
        return False

    return True


# =============================================================================
# Model Endpoint Tests
# =============================================================================


def test_detection() -> bool:
    """Test object detection endpoint."""
    print_header('OBJECT DETECTION')

    if not TEST_IMAGE.exists():
        print_test('Detection', 'SKIP', f'Test image not found: {TEST_IMAGE}')
        return False

    with open(TEST_IMAGE, 'rb') as f:
        files = {'image': f}
        data = test_endpoint('Single Detection', 'POST', f'{API_BASE}/detect', files=files)

    if data and 'detections' in data:
        num_dets = len(data['detections'])
        print_test('Detection Result', 'PASS', f'{num_dets} objects detected')
        if num_dets > 0:
            print(f'       First detection: {data["detections"][0]}')
        return True

    return False


def test_faces() -> bool:
    """Test face detection and recognition."""
    print_header('FACE DETECTION & RECOGNITION')

    if not TEST_IMAGE.exists():
        print_test('Face Tests', 'SKIP', f'Test image not found: {TEST_IMAGE}')
        return False

    # Face detection
    with open(TEST_IMAGE, 'rb') as f:
        files = {'image': f}
        data = test_endpoint('Face Detection', 'POST', f'{API_BASE}/faces/detect', files=files)

    if data and 'num_faces' in data:
        print_test('Face Detection Result', 'PASS', f'{data["num_faces"]} faces detected')
    else:
        return False

    # Face recognition (detection + embeddings)
    with open(TEST_IMAGE, 'rb') as f:
        files = {'image': f}
        data = test_endpoint('Face Recognition', 'POST', f'{API_BASE}/faces/recognize', files=files)

    if data and 'embeddings' in data:
        num_embeddings = len(data['embeddings'])
        print_test('Face Embeddings', 'PASS', f'{num_embeddings} embeddings (512-dim each)')
        return True

    return False


def test_embeddings() -> bool:
    """Test CLIP embedding endpoints."""
    print_header('CLIP EMBEDDINGS')

    if not TEST_IMAGE.exists():
        print_test('Embedding Tests', 'SKIP', f'Test image not found: {TEST_IMAGE}')
        return False

    # Image embedding
    with open(TEST_IMAGE, 'rb') as f:
        files = {'image': f}
        data = test_endpoint('Image Embedding', 'POST', f'{API_BASE}/embed/image', files=files)

    if data and 'embedding' in data:
        emb_len = len(data['embedding'])
        print_test('Image Embedding Result', 'PASS', f'{emb_len}-dim vector')
    else:
        return False

    # Text embedding
    data = test_endpoint(
        'Text Embedding', 'POST', f'{API_BASE}/embed/text', json={'text': 'soccer player'}
    )

    if data and 'embedding' in data:
        emb_len = len(data['embedding'])
        print_test('Text Embedding Result', 'PASS', f'{emb_len}-dim vector')
        return True

    return False


def test_ocr() -> bool:
    """Test OCR endpoint."""
    print_header('OCR (TEXT EXTRACTION)')

    # Try to find an image with text
    ocr_images = list(Path('test_images').glob('**/testocr*.jpg'))
    if not ocr_images:
        ocr_images = [TEST_IMAGE]

    test_img = ocr_images[0]
    if not test_img.exists():
        print_test('OCR', 'SKIP', 'No test image found')
        return False

    with open(test_img, 'rb') as f:
        files = {'image': f}
        data = test_endpoint('OCR Prediction', 'POST', f'{API_BASE}/ocr/predict', files=files)

    if data and 'num_texts' in data:
        num_texts = data['num_texts']
        print_test('OCR Result', 'PASS', f'{num_texts} text regions detected')
        if num_texts > 0 and 'texts' in data:
            print(f'       First text: "{data["texts"][0]}"')
        return True

    return False


def test_analyze() -> bool:
    """Test combined analysis endpoint."""
    print_header('COMBINED ANALYSIS')

    if not TEST_IMAGE.exists():
        print_test('Analyze', 'SKIP', f'Test image not found: {TEST_IMAGE}')
        return False

    with open(TEST_IMAGE, 'rb') as f:
        files = {'image': f}
        data = test_endpoint('Full Analyze', 'POST', f'{API_BASE}/analyze', files=files)

    if data:
        summary = []
        if 'num_detections' in data:
            summary.append(f'{data["num_detections"]} objects')
        if 'num_faces' in data:
            summary.append(f'{data["num_faces"]} faces')
        if 'has_text' in data:
            summary.append(f'text={"yes" if data["has_text"] else "no"}')
        if 'embedding_norm' in data and data['embedding_norm'] is not None:
            summary.append(f'embedding_norm={data["embedding_norm"]:.3f}')

        print_test('Analyze Result', 'PASS', ', '.join(summary))
        return True

    return False


# =============================================================================
# Ingest Pipeline Tests
# =============================================================================


def test_ingest_single() -> bool:
    """Test single image ingestion."""
    print_header('SINGLE IMAGE INGEST')

    if not TEST_IMAGE.exists():
        print_test('Ingest', 'SKIP', f'Test image not found: {TEST_IMAGE}')
        return False

    with open(TEST_IMAGE, 'rb') as f:
        files = {'image': f}
        params = {'image_id': 'test_ingest_001'}
        data = test_endpoint(
            'Single Image Ingest', 'POST', f'{API_BASE}/ingest', files=files, params=params
        )

    if data and data.get('status') in ['success', 'duplicate']:
        indexed = data.get('indexed', {})
        summary = ', '.join(f'{k}={v}' for k, v in indexed.items() if isinstance(v, int))
        print_test('Ingest Result', 'PASS', f'Status: {data["status"]}, indexed: {summary}')
        return True

    return False


def test_ingest_directory(skip: bool = False) -> bool:
    """Test directory ingestion (slow)."""
    print_header('DIRECTORY INGEST')

    if skip:
        print_test('Directory Ingest', 'SKIP', 'Skipped by user')
        return True

    if not INGEST_DIR.exists():
        print_test('Directory Ingest', 'SKIP', f'Directory not found: {INGEST_DIR}')
        return True

    num_images = len(list(INGEST_DIR.glob('*.jpg')))
    print(f'  Ingesting {num_images} images from {INGEST_DIR}...')

    start_time = time.time()
    data = test_endpoint(
        'Directory Ingest',
        'POST',
        f'{API_BASE}/ingest/directory?directory=/app/{INGEST_DIR}&batch_size=32',
        timeout=120,  # Allow 2 minutes for batch processing
    )

    if data and 'total_processed' in data:
        elapsed = time.time() - start_time
        processed = data['total_processed']
        rate = processed / elapsed if elapsed > 0 else 0
        print_test(
            'Ingest Result',
            'PASS',
            f'{processed} images, {rate:.1f} images/sec, {elapsed:.1f}s total',
        )
        return True

    return False


# =============================================================================
# Search & Query Tests
# =============================================================================


def test_opensearch_indexes() -> bool:
    """Test OpenSearch index status."""
    print_header('OPENSEARCH INDEXES')

    response = requests.get(f'{OPENSEARCH_BASE}/_cat/indices?v&format=json', timeout=10)
    if response.status_code == 200:
        indexes = response.json()
        visual_indexes = [idx for idx in indexes if idx['index'].startswith('visual_search_')]

        if not visual_indexes:
            print_test('OpenSearch Indexes', 'FAIL', 'No visual_search_* indexes found')
            return False

        for idx in visual_indexes:
            name = idx['index']
            doc_count = idx['docs.count']
            size = idx['store.size']
            print_test(f'Index: {name}', 'PASS', f'{doc_count} docs, {size}')
        return True

    print_test('OpenSearch Indexes', 'FAIL', 'Could not list indexes')
    return False


def test_query_stats() -> bool:
    """Test query stats endpoint."""
    print_header('QUERY STATISTICS')

    data = test_endpoint('Query Stats', 'GET', f'{API_BASE}/query/stats')

    if data and 'indexes' in data:
        indexes = data['indexes']
        if isinstance(indexes, list):
            for idx_data in indexes:
                if isinstance(idx_data, dict) and idx_data.get('exists'):
                    idx_name = idx_data.get('name', 'unknown')
                    doc_count = idx_data.get('doc_count', 0)
                    size_human = idx_data.get('size_human', '0 B')
                    print_test(f'Index: {idx_name}', 'PASS', f'{doc_count} docs, {size_human}')
        return True

    return False


def test_image_search() -> bool:
    """Test image similarity search."""
    print_header('IMAGE SIMILARITY SEARCH')

    if not TEST_IMAGE.exists():
        print_test('Image Search', 'SKIP', f'Test image not found: {TEST_IMAGE}')
        return False

    with open(TEST_IMAGE, 'rb') as f:
        files = {'image': f}
        data = test_endpoint(
            'Image Search', 'POST', f'{API_BASE}/search/image?top_k=5', files=files
        )

    if data and 'results' in data:
        num_results = len(data['results'])
        print_test('Search Results', 'PASS', f'{num_results} similar images found')
        if num_results > 0:
            top_result = data['results'][0]
            print(f'       Top result: score={top_result.get("score", 0):.3f}')
        return True

    return False


def test_text_search() -> bool:
    """Test text-to-image search."""
    print_header('TEXT-TO-IMAGE SEARCH')

    data = test_endpoint(
        'Text Search', 'POST', f'{API_BASE}/search/text?text=soccer player&top_k=5'
    )

    if data and 'results' in data:
        num_results = len(data['results'])
        print_test('Text Search Results', 'PASS', f'{num_results} matching images found')
        return True

    return False


# =============================================================================
# Main Test Runner
# =============================================================================


def run_all_tests(skip_ingest: bool = False) -> bool:
    """Run all tests and report results."""
    print('\n' + '=' * 80)
    print('TRITON API - COMPREHENSIVE SYSTEM TEST SUITE'.center(80))
    print('=' * 80)

    # Phase 0: Clear OpenSearch data for fresh testing
    clear_opensearch_data()

    # Phase 1: Health checks (critical)
    if not test_health_checks():
        print('\n❌ Services not healthy - stopping tests')
        return False

    # Phase 2: Model endpoints
    test_detection()
    test_faces()
    test_embeddings()
    test_ocr()
    test_analyze()

    # Phase 3: Ingest pipeline
    test_ingest_single()
    test_ingest_directory(skip=skip_ingest)

    # Phase 4: OpenSearch & queries
    test_opensearch_indexes()
    test_query_stats()
    test_image_search()
    test_text_search()

    # Print summary
    print_header('TEST SUMMARY')
    total = results['passed'] + results['failed'] + results['skipped']
    pass_pct = (results['passed'] / total * 100) if total > 0 else 0

    print(f'\n  Total Tests:  {total}')
    print(f'  ✓ Passed:     {results["passed"]} ({pass_pct:.1f}%)')
    print(f'  ✗ Failed:     {results["failed"]}')
    print(f'  ○ Skipped:    {results["skipped"]}')

    if results['failed'] == 0:
        print('\n✅ All tests passed!')
        return True

    print('\n❌ Some tests failed. Review output above for details.')
    return False


# =============================================================================
# Entry Point
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Comprehensive Triton API test suite')
    parser.add_argument(
        '--skip-ingest',
        action='store_true',
        help='Skip directory ingestion (slow)',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON',
    )
    args = parser.parse_args()

    success = run_all_tests(skip_ingest=args.skip_ingest)

    if args.json:
        print('\n' + json.dumps(results, indent=2))

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
