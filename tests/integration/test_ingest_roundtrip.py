"""Ingest -> crops/status round trip through the real app.

Ingests two images through ``POST /curation/ingest/batch`` against a
fake OpenSearch + a scripted detector/PE-encoder, then asserts
``GET /curation/crops`` lists both items with the quality fields
(``crop_area_norm``, ``blur_lap_var``, ``blur_lap_ratio``) populated,
and ``GET /curation/ingest/status`` reflects the new images.

Per plan §6.0 house rule, this fakes the OpenSearch/Triton I/O boundary
rather than standing up a live stack — see
``tests/integration/test_ingest_occ.py`` for the same convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from integration.ingest_fakes import (
    FakeOpenSearch,
    FakeTritonPool,
    curation_app,
    jpeg_bytes as _jpeg_bytes,
)


if TYPE_CHECKING:
    from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


@pytest.fixture
def fake_opensearch() -> FakeOpenSearch:
    return FakeOpenSearch()


@pytest.fixture
def fake_triton() -> FakeTritonPool:
    return FakeTritonPool()


@pytest.fixture
def client(
    fake_opensearch: FakeOpenSearch,
    fake_triton: FakeTritonPool,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    with curation_app(fake_opensearch, fake_triton, monkeypatch) as c:
        yield c


def test_ingest_batch_then_crops_and_status(
    client: TestClient, fake_opensearch: FakeOpenSearch, fake_triton: FakeTritonPool
) -> None:
    body = {
        'items': [
            {'path': '/tmp/roundtrip_a.jpg', 'source': 'roundtrip_test'},
            {'path': '/tmp/roundtrip_b.jpg', 'source': 'roundtrip_test'},
        ]
    }
    # Write real files so the router's Path(...).read_bytes() succeeds.
    from pathlib import Path

    Path('/tmp/roundtrip_a.jpg').write_bytes(_jpeg_bytes(1))
    Path('/tmp/roundtrip_b.jpg').write_bytes(_jpeg_bytes(2))

    resp = client.post('/curation/ingest/batch', json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload['summary']['successful'] == 2
    assert payload['summary']['crops_indexed'] == 2

    crops_resp = client.get('/curation/crops')
    assert crops_resp.status_code == 200, crops_resp.text
    crops_payload = crops_resp.json()
    items = crops_payload.get('items') or crops_payload.get('crops') or crops_payload
    assert isinstance(items, list)
    assert len(items) == 2
    for item in items:
        # crops.py's wire model surfaces crop_area_norm/crop_rank_in_image/
        # blur_lap_ratio (not blur_lap_var, which is diagnostic-only and
        # excluded from that response by design) -- checked against the
        # response here, and against the raw OpenSearch doc below.
        assert item.get('crop_area_norm') is not None
        assert item.get('blur_lap_ratio') is not None
        assert item.get('crop_rank_in_image') is not None

    assert len(fake_opensearch.items) == 2
    for doc in fake_opensearch.items.values():
        assert doc.get('blur_lap_var') is not None
        assert doc.get('blur_lap_ratio') is not None
        assert doc.get('crop_area_norm') is not None
        assert doc.get('pe_embedding') is not None

    status_resp = client.get('/curation/ingest/status')
    assert status_resp.status_code == 200, status_resp.text
    status_payload = status_resp.json()
    assert status_payload['total'] == 2

    # G11 guard, end to end through the router: two images cost exactly
    # one batched Triton round-trip, not two single-image ones.
    assert fake_triton.batch_sizes == [2]


def test_ingest_batch_imports_companion_labels(
    client: TestClient, fake_opensearch: FakeOpenSearch
) -> None:
    """G12 guard: images + paired ground-truth YOLO labels in one call."""
    from pathlib import Path

    image_path = Path('/tmp/roundtrip_labeled.jpg')
    image_path.write_bytes(_jpeg_bytes(7))
    label_path = Path('/tmp/roundtrip_labeled.txt')
    # Same region the fake detector proposes (norm box 0.1,0.1..0.5,0.5
    # of the letterboxed square maps to roughly the upper-left quadrant).
    label_path.write_text('0 0.3 0.3 0.4 0.4\n')

    resp = client.post(
        '/curation/ingest/batch',
        json={
            'items': [
                {
                    'path': str(image_path),
                    'source': 'roundtrip_test',
                    'label_txt_path': str(label_path),
                }
            ],
            'label_source': 'ground_truth',
            'detect_mismatches': True,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload['summary']['successful'] == 1
    assert payload['summary']['labels_imported'] == 1
    assert len(fake_opensearch.labels) == 1
    [label_doc] = list(fake_opensearch.labels.values())
    assert label_doc['class_id'] == 0
    assert label_doc['label_source'] == 'ground_truth'
